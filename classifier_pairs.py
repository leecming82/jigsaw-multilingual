"""
Modification of baseline classifier that
combines raw and english translations into a single token stream per sample
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
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import roc_auc_score
from apex import amp
from tqdm import trange
from preprocessor import get_translation_pair_from_csv, get_id_text_from_test_csv
from torch_helpers import save_model
from swa import SWA

RUN_NAME = 'xlmr-pair'  # used when writing outputs
PREDICT = True
USE_SWA = False
SAVE_MODEL = False
USE_AMP = True
PRETRAINED_MODEL = 'xlm-roberta-large'
TRAIN_SAMPLE_FRAC = .05  # what % of training data to use
TRAIN_CSV_PATH = 'data/translated_2018/combined.csv'
TEST_CSV_PATH = 'data/test_en.csv'
VAL_CSV_PATH = 'data/validation_en.csv'
PSEUDO_CSV_PATH = 'data/submissions/test9440.csv'
MODEL_SAVE_DIR = 'models/{}'.format(RUN_NAME)
MAX_CORES = 24  # limit MP calls to use this # cores at most
BASE_MODEL_OUTPUT_DIM = 1024  # hidden layer dimensions
NUM_OUTPUTS = 1
MAX_SEQ_LEN = 100  # for EACH comment of the raw/english comments
NUM_EPOCHS = 6
BATCH_SIZE = 16
ACCUM_FOR = 2
LR = 1e-5
SWA_START_STEP = 2000  # counts only optimizer steps - so note if ACCUM_FOR > 1
SWA_FREQ = 20

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
        hidden_states = self.base_model(x)[0][:, 0, :]
        # hidden_states = hidden_states.permute(0, 2, 1)
        # cnn_states = self.cnn(hidden_states)
        # cnn_states = cnn_states.permute(0, 2, 1)
        # logits, _ = torch.max(cnn_states, 1)
        logits = self.fc(hidden_states)
        prob = torch.nn.Sigmoid()(logits)
        return prob


def train(model, train_tuple, loss_fn, opt, curr_epoch):
    """ Train """
    # Shuffle train indices for current epoch, batching
    all_features, all_labels, all_ids = train_tuple
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

            batch_features = torch.tensor(train_features[batch_idx_start:batch_idx_end]).long().cuda()
            batch_labels = torch.tensor(train_labels[batch_idx_start:batch_idx_end]).float().cuda().unsqueeze(-1)

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


def predict_evaluate(model, data_tuple, epoch, score=False):
    # Evaluate validation AUC
    val_score = None
    val_preds = []
    model.eval()
    with torch.no_grad():
        for batch_idx_start in range(0, len(data_tuple[-1]), BATCH_SIZE):
            batch_idx_end = min(batch_idx_start + BATCH_SIZE, len(data_tuple[-1]))
            batch_features = torch.tensor(data_tuple[0][batch_idx_start:batch_idx_end]).long().cuda()
            batch_preds = model(batch_features)
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

    if USE_SWA:
        opt = SWA(opt, swa_start=SWA_START_STEP, swa_freq=SWA_FREQ)

    if USE_AMP:
        amp.register_float_function(torch, 'sigmoid')
        classifier, opt = amp.initialize(classifier, opt, opt_level='O1', verbosity=0)

    list_raw_auc = []

    current_tuple = train_tuple
    for curr_epoch in range(NUM_EPOCHS):
        if curr_epoch >= NUM_EPOCHS // 2:
            current_tuple = val_tuple
        train(classifier, current_tuple, loss_fn, opt, curr_epoch)

        epoch_raw_auc = predict_evaluate(classifier, val_tuple, curr_epoch, score=True)
        print('Epoch {} - Raw: {:.4f}'.format(curr_epoch, epoch_raw_auc))
        list_raw_auc.append(epoch_raw_auc)

        if PREDICT:
            predict_evaluate(classifier, test_tuple, curr_epoch)

        if SAVE_MODEL:
            save_model(os.path.join(MODEL_SAVE_DIR, str(curr_epoch)), classifier, pretrained_config, tokenizer)

    if USE_SWA:
        opt.swap_swa_sgd()
        epoch_raw_auc = predict_evaluate(classifier, val_tuple, 'SWA', score=True)
        print('SWA - Raw: {:.4f}'.format(epoch_raw_auc))
        list_raw_auc.append(epoch_raw_auc)
        if SAVE_MODEL:
            save_model(os.path.join(MODEL_SAVE_DIR, 'SWA'), classifier, pretrained_config, tokenizer)

    with np.printoptions(precision=4, suppress=True):
        print(np.array(list_raw_auc))

    pd.DataFrame({'val_auc': list_raw_auc}).to_csv('data/outputs/results/{}.csv'.format(RUN_NAME), index=False)


if __name__ == '__main__':
    start_time = time.time()

    # # Load train, validation, and pseudo-label data
    (train_ids, train_raw_strings, train_en_strings, train_labels) = \
        get_translation_pair_from_csv(TRAIN_CSV_PATH,
                                      raw_text_col='comment_text',
                                      en_text_col='comment_text_en',
                                      sample_frac=TRAIN_SAMPLE_FRAC)

    (val_ids, val_raw_strings, val_en_strings, val_labels) = \
        get_translation_pair_from_csv(VAL_CSV_PATH,
                                      raw_text_col='comment_text',
                                      en_text_col='comment_text_en')

    test_ids, test_raw_strings = get_id_text_from_test_csv(TEST_CSV_PATH, text_col='content')
    _, test_en_strings = get_id_text_from_test_csv(TEST_CSV_PATH, text_col='content_en')

    # use MP to batch encode the raw feature strings into Bert token IDs
    tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_MODEL)
    encode_partial = partial(tokenizer.encode,
                             max_length=MAX_SEQ_LEN,
                             add_special_tokens=True)

    print('Encoding raw strings into model-specific tokens')
    with mp.Pool(MAX_CORES) as p:
        train_raw_features = list(p.map(encode_partial, train_raw_strings))
        train_en_features = list(p.map(encode_partial, train_en_strings))
        train_features = pad_sequences([x + y for (x, y) in zip(train_raw_features, train_en_features)],
                                       maxlen=MAX_SEQ_LEN * 2,
                                       padding='post',
                                       truncating='post',
                                       value=tokenizer.pad_token_id)

        val_raw_features = list(p.map(encode_partial, val_raw_strings))
        val_en_features = list(p.map(encode_partial, val_en_strings))
        val_features = pad_sequences([x + y for (x, y) in zip(val_raw_features, val_en_features)],
                                     maxlen=MAX_SEQ_LEN * 2,
                                     padding='post',
                                     truncating='post',
                                     value=tokenizer.pad_token_id)

        test_raw_features = list(p.map(encode_partial, test_raw_strings))
        test_en_features = list(p.map(encode_partial, test_en_strings))
        test_features = pad_sequences([x + y for (x, y) in zip(test_raw_features, test_en_features)],
                                      maxlen=MAX_SEQ_LEN * 2,
                                      padding='post',
                                      truncating='post',
                                      value=tokenizer.pad_token_id)

    print(train_features[0])
    print(val_features[0])
    print(test_features[0])

    print('Train size: {}, val size: {}'.format(len(train_ids), len(val_ids)))
    print('Train positives: {}, train negatives: {}'.format(train_labels[train_labels == 1].shape,
                                                            train_labels[train_labels == 0].shape))

    main_driver([train_features, train_labels, train_ids],
                [val_features, val_labels, val_ids],
                [test_features, test_ids],
                tokenizer)

    print('Elapsed time: {}'.format(time.time() - start_time))
