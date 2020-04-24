"""
Modification of baseline classifier to apply mixup (at the embedding level)
"""
import os
import time
from itertools import starmap
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
from preprocessor import get_id_text_label_from_csv, get_id_text_from_test_csv
from torch_helpers import save_model
from swa import SWA

RUN_NAME = '9509es_mixup'  # used when writing outputs
PREDICT = True
USE_PSEUDO = False
PRETRAINED_MODEL = 'mrm8488/distill-bert-base-spanish-wwm-cased-finetuned-spa-squad2-es'
TRAIN_SAMPLE_FRAC = 1.  # what % of training data to use
TRAIN_CSV_PATH = 'data/es_all.csv'
TEST_CSV_PATH = 'data/es_test.csv'
PSEUDO_CSV_PATH = 'data/submissions/test9452.csv'
VAL_CSV_PATH = 'data/validation.csv'
MAX_CORES = 24  # limit MP calls to use this # cores at most
BASE_MODEL_OUTPUT_DIM = 768  # hidden layer dimensions
NUM_OUTPUTS = 1
MAX_SEQ_LEN = 200  # max sequence length for input strings: gets padded/truncated
NUM_EPOCHS = 6
BATCH_SIZE = 64
ACCUM_FOR = 1
LR = 1e-5
MIXUP_RATE = 1.

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


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
        hidden_states = self.base_model(inputs_embeds=x)[0]
        # hidden_states = hidden_states.permute(0, 2, 1)
        # cnn_states = self.cnn(hidden_states)
        # cnn_states = cnn_states.permute(0, 2, 1)
        # logits, _ = torch.max(cnn_states, 1)
        hidden = hidden_states[:, 0, :]
        logits = self.fc(hidden)
        prob = torch.nn.Sigmoid()(logits)
        return prob


def train(model, train_tuple, loss_fn, opt, curr_epoch):
    """ Train """
    # Shuffle train indices for current epoch, batching
    all_features, all_labels, all_ids = train_tuple
    train_indices = list(range(len(all_labels)))
    copy_train_indices = np.copy(train_indices)

    shuffle(train_indices)
    shuffle(copy_train_indices)

    train_features_left, train_labels_left = all_features[train_indices], all_labels[train_indices]
    train_features_right, train_labels_right = all_features[copy_train_indices], all_labels[copy_train_indices]

    model.train()
    iter = 0
    running_total_loss = 0
    with trange(0, len(train_indices), BATCH_SIZE,
                desc='Epoch {}'.format(curr_epoch)) as t:
        for batch_idx_start in t:
            iter += 1
            batch_idx_end = min(batch_idx_start + BATCH_SIZE, len(train_indices))

            batch_features_left = torch.tensor(train_features_left[batch_idx_start:batch_idx_end]).cuda()
            embed_left = model.base_model.embeddings(batch_features_left)

            batch_features_right = torch.tensor(train_features_right[batch_idx_start:batch_idx_end]).cuda()
            embed_right = model.base_model.embeddings(batch_features_right)

            p = np.random.beta(MIXUP_RATE, MIXUP_RATE, size=(1, 1))
            combined_labels = p * train_labels_left[batch_idx_start:batch_idx_end] + \
                              (1 - p) * train_labels_right[batch_idx_start:batch_idx_end]
            p = torch.tensor(p).reshape(1, 1, 1).float().cuda()
            combined_embed = p * embed_left + (1 - p) * embed_right
            batch_labels = torch.tensor(combined_labels).float().cuda().permute(1, 0)

            preds = model(combined_embed)
            loss = loss_fn(preds, batch_labels)
            loss = loss / ACCUM_FOR

            with amp.scale_loss(loss, opt) as scaled_loss:
                scaled_loss.backward()

            running_total_loss += loss.detach().cpu().numpy()
            t.set_postfix(loss=running_total_loss / iter)

            if iter % ACCUM_FOR == 0:
                opt.step()
                opt.zero_grad()


def predict_evaluate(model, data_tuple, epoch, score=False):
    # Evaluate validation AUC
    val_score = None
    val_preds = []
    model.eval()
    with torch.no_grad():
        for batch_idx_start in range(0, len(data_tuple[-1]), BATCH_SIZE):
            batch_idx_end = min(batch_idx_start + BATCH_SIZE, len(data_tuple[-1]))
            batch_features = torch.tensor(data_tuple[0][batch_idx_start:batch_idx_end]).cuda()
            embed = model.base_model.embeddings(batch_features)
            batch_preds = model(embed)
            val_preds.append(batch_preds.cpu().numpy().squeeze())

    val_preds = np.concatenate(val_preds)
    if score:
        val_score = roc_auc_score(data_tuple[1], val_preds)

    save_folder = 'validation' if score else 'test'
    pd.DataFrame({'id': data_tuple[-1], 'toxic': val_preds}) \
        .to_csv('data/outputs/{}/{}_{}.csv'.format(save_folder,
                                                   RUN_NAME,
                                                   epoch),
                index=False)

    return val_score


def main_driver(train_tuple, val_tuple, test_tuple, tokenizer):
    pretrained_config = AutoConfig.from_pretrained(PRETRAINED_MODEL,
                                                   output_hidden_states=True)
    pretrained_base = AutoModel.from_pretrained(PRETRAINED_MODEL, config=pretrained_config).cuda()
    classifier = ClassifierHead(pretrained_base).cuda()
    loss_fn = torch.nn.BCELoss()
    opt = torch.optim.Adam(classifier.parameters(), lr=LR)

    amp.register_float_function(torch, 'sigmoid')
    classifier, opt = amp.initialize(classifier, opt, opt_level='O1', verbosity=0)

    list_raw_auc = []

    current_tuple = train_tuple
    for curr_epoch in range(NUM_EPOCHS):
        if curr_epoch == NUM_EPOCHS // 2:
            current_tuple = val_tuple
        train(classifier, current_tuple, loss_fn, opt, curr_epoch)

        epoch_raw_auc = predict_evaluate(classifier, val_tuple, curr_epoch, score=True)
        print('Epoch {} - Raw: {:.4f}'.format(curr_epoch, epoch_raw_auc))
        list_raw_auc.append(epoch_raw_auc)

        if PREDICT:
            predict_evaluate(classifier, test_tuple, curr_epoch)

    with np.printoptions(precision=4, suppress=True):
        print(np.array(list_raw_auc))

    pd.DataFrame({'val_auc': list_raw_auc}).to_csv('data/outputs/results/{}.csv'.format(RUN_NAME), index=False)


if __name__ == '__main__':
    def cln(x):
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

    val_ids, val_strings, val_labels = get_id_text_label_from_csv(VAL_CSV_PATH, text_col='comment_text', lang='es')
    val_strings = [cln(x) for x in val_strings]

    test_ids, test_strings = get_id_text_from_test_csv(TEST_CSV_PATH, text_col='comment_text')
    test_strings = [cln(x) for x in test_strings]

    pseudo_ids = []
    if USE_PSEUDO:
        pseudo_ids, pseudo_strings, pseudo_labels = get_id_text_label_from_csv(PSEUDO_CSV_PATH,
                                                                               text_col='content')

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
        if USE_PSEUDO:
            pseudo_features = np.array(p.map(encode_partial, pseudo_strings))

    if USE_PSEUDO:
        train_features = np.concatenate([train_features, pseudo_features])
        train_labels = np.concatenate([train_labels, pseudo_labels])
        train_ids = np.concatenate([train_ids, pseudo_ids])

    print('Train size: {}, val size: {}, pseudo size: {}'.format(len(train_ids), len(val_ids), len(pseudo_ids)))
    print('Train positives: {}, train negatives: {}'.format(train_labels[train_labels == 1].shape,
                                                            train_labels[train_labels == 0].shape))

    main_driver([train_features, train_labels, train_ids],
                [val_features, val_labels, val_ids],
                [test_features, test_ids],
                tokenizer)

    print('Elapsed time: {}'.format(time.time() - start_time))
