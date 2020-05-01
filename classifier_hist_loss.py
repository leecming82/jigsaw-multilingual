"""
Converts single prob point estimate to bins - histogram loss
"""
import os
import time
from random import shuffle
from functools import partial
import multiprocessing as mp
import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel, AutoConfig
from sklearn.metrics import roc_auc_score
from apex import amp
from tqdm import trange
from preprocessor import get_id_text_label_from_csv, generate_target_dist, get_id_text_from_test_csv

RUN_NAME = '9536it_xxl_uncased_hist'  # added as prefix to file outputs
PREDICT = True
USE_VAL_LANG = 'it'
USE_AMP = True
PRETRAINED_MODEL = 'dbmdz/bert-base-italian-xxl-uncased'
TRAIN_SAMPLE_FRAC = 1.  # what % of training data to use
TRAIN_CSV_PATH = 'data/it_all.csv'
TEST_CSV_PATH = 'data/it_test.csv'
VAL_CSV_PATH = 'data/val_preds.csv'
MAX_CORES = 24  # limit MP calls to use this # cores at most
BASE_MODEL_OUTPUT_DIM = 768  # hidden layer dimensions
NUM_OUTPUTS = 10
MAX_SEQ_LEN = 200  # max sequence length for input strings: gets padded/truncated
NUM_EPOCHS = 6  # Half trained using train, half on val (+ PL)
BATCH_SIZE = 64
ACCUM_FOR = 1
LR = 1e-5
LOW = -0.2
HIGH = 1.2
NUM_BINS = 10

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'


class ClassifierHead(torch.nn.Module):
    """
    Bert base with a Linear layer plopped on top of it
    - connects the max pool of the last hidden layer with the FC
    """
    def __init__(self, base_model):
        super(ClassifierHead, self).__init__()
        self.base_model = base_model
        self.cnn = torch.nn.Conv1d(BASE_MODEL_OUTPUT_DIM, NUM_OUTPUTS, kernel_size=1)
        self.fc = torch.nn.Linear(BASE_MODEL_OUTPUT_DIM, NUM_OUTPUTS)

    def forward(self, x):
        hidden_states = self.base_model(x)[0]
        logits = self.fc(hidden_states[:, 0, :])
        prob = torch.nn.Softmax(dim=-1)(logits)
        return prob


def train(model, train_tuple, loss_fn, opt, curr_epoch):
    """ Train """
    # Shuffle train indices for current epoch, batching
    all_features, all_labels, _ = train_tuple
    train_indices = list(range(len(all_labels)))

    shuffle(train_indices)
    train_features = all_features[train_indices]
    train_labels = all_labels[train_indices]

    model.train()
    iter = 0
    running_total_loss = 0
    with trange(0, len(train_indices), BATCH_SIZE,
                desc='Epoch {}'.format(curr_epoch)) as t:
        for batch_idx_start in t:
            iter += 1
            batch_idx_end = min(batch_idx_start + BATCH_SIZE, len(train_indices))

            batch_features = torch.tensor(train_features[batch_idx_start:batch_idx_end]).cuda()
            batch_labels = torch.tensor(train_labels[batch_idx_start:batch_idx_end]).float().cuda()

            preds = model(batch_features)
            loss = loss_fn(preds, batch_labels)
            loss = loss / ACCUM_FOR

            if USE_AMP:
                with amp.scale_loss(loss, opt) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            running_total_loss += loss.detach().cpu().numpy()
            t.set_postfix(loss=running_total_loss / iter)

            if iter % ACCUM_FOR == 0:
                opt.step()
                opt.zero_grad()


def predict_evaluate(model, data_tuple, supports, epoch, score=False):
    """
    Make predictions against either val or test set
    Saves output to csv in data/outputs/test or data/outputs/validation
    """
    val_score = None
    val_preds = []
    model.eval()
    with torch.no_grad():
        for batch_idx_start in range(0, len(data_tuple[-1]), BATCH_SIZE):
            batch_idx_end = min(batch_idx_start + BATCH_SIZE, len(data_tuple[-1]))
            batch_features = torch.tensor(data_tuple[0][batch_idx_start:batch_idx_end]).cuda()
            batch_preds = model(batch_features)
            batch_preds = np.dot(batch_preds.cpu(), supports)
            val_preds.append(batch_preds)

    val_preds = np.concatenate(val_preds)
    if score:
        val_score = roc_auc_score(np.round(data_tuple[2]), val_preds)

    save_folder = 'validation' if score else 'test'
    pd.DataFrame({'id': data_tuple[-1], 'toxic': val_preds}) \
        .to_csv('data/outputs/{}/{}_{}.csv'.format(save_folder,
                                                   RUN_NAME,
                                                   epoch),
                index=False)

    return val_score


def main_driver(train_tuple, val_tuple, test_tuple, supports, tokenizer):
    pretrained_config = AutoConfig.from_pretrained(PRETRAINED_MODEL,
                                                   output_hidden_states=True)
    pretrained_base = AutoModel.from_pretrained(PRETRAINED_MODEL, config=pretrained_config).cuda()
    classifier = ClassifierHead(pretrained_base).cuda()
    loss_fn = torch.nn.BCELoss()
    opt = torch.optim.Adam(classifier.parameters(), lr=LR)

    if USE_AMP:
        amp.register_float_function(torch, 'sigmoid')
        classifier, opt = amp.initialize(classifier, opt, opt_level='O1', verbosity=0)

    list_auc = []

    current_tuple = train_tuple
    for curr_epoch in range(NUM_EPOCHS):
        # Switch from toxic-2018 to the current mixed language dataset, halfway thru
        if curr_epoch == NUM_EPOCHS // 2:
            current_tuple = [val_tuple[0], val_tuple[1], val_tuple[3]]
        train(classifier, current_tuple, loss_fn, opt, curr_epoch)

        epoch_raw_auc = predict_evaluate(classifier, val_tuple, supports, curr_epoch, score=True)
        print('Epoch {} - Raw: {:.4f}'.format(curr_epoch, epoch_raw_auc))
        list_auc.append(epoch_raw_auc)

        if PREDICT:
            predict_evaluate(classifier, test_tuple, supports, curr_epoch)

    with np.printoptions(precision=4, suppress=True):
        print(np.array(list_auc))


if __name__ == '__main__':
    def cln(x):  # Truncates adjacent whitespaces to single whitespace
        return ' '.join(x.split())

    start_time = time.time()
    print(RUN_NAME)

    # Load train, validation, and pseudo-label data
    train_ids, train_strings, train_labels = get_id_text_label_from_csv(TRAIN_CSV_PATH,
                                                                        text_col='comment_text',
                                                                        sample_frac=TRAIN_SAMPLE_FRAC)
    print(train_strings[0])
    train_strings = [cln(x) for x in train_strings]
    print(train_strings[0])

    val_ids, val_strings, val_labels = get_id_text_label_from_csv(VAL_CSV_PATH,
                                                                  text_col='comment_text',
                                                                  lang=USE_VAL_LANG)
    val_strings = [cln(x) for x in val_strings]

    test_ids, test_strings = get_id_text_from_test_csv(TEST_CSV_PATH, text_col='comment_text')
    test_strings = [cln(x) for x in test_strings]

    # use MP to batch encode the raw feature strings into Bert token IDs
    tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_MODEL)
    encode_partial = partial(tokenizer.encode,
                             max_length=MAX_SEQ_LEN,
                             pad_to_max_length=True,
                             add_special_tokens=True)
    print('Encoding raw strings into model-specific tokens')
    with mp.Pool(MAX_CORES) as p:
        train_features = np.array(p.map(encode_partial, train_strings))
        val_features = np.array(p.map(encode_partial, val_strings))
        test_features = np.array(p.map(encode_partial, test_strings))

    generate_target_partial = partial(generate_target_dist, num_bins=NUM_BINS, low=LOW, high=HIGH)
    with mp.Pool(MAX_CORES) as p:
        train_hist_labels = np.stack([x[1] for x in p.map(generate_target_partial, train_labels)])
        val_hist_labels = np.stack([x[1] for x in p.map(generate_target_partial, val_labels)])
    supports = generate_target_partial(-1)[0]
    print(train_labels.shape)

    print('Train size: {}, val size: {}, test size: {}'.format(len(train_ids), len(val_ids), len(test_ids)))
    print('Train positives: {}, train negatives: {}'.format(train_labels[train_labels == 1].shape,
                                                            train_labels[train_labels == 0].shape))

    main_driver([train_features, train_hist_labels, train_ids],
                [val_features, val_hist_labels, val_labels, val_ids],
                [test_features, test_ids],
                supports,
                tokenizer)

    print('Elapsed time: {}'.format(time.time() - start_time))
