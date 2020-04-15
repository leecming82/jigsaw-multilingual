"""
Baseline PyTorch classifier for Jigsaw Multilingual
- Assumes two separate train and val sets (i.e., no need for k-folds)
- Splits epochs between training the train set and the val set (i.e., 0.5 NUM_EPOCHS each)
"""
import os
import time
import pandas as pd
from itertools import starmap
from random import shuffle
from functools import partial
import multiprocessing as mp
import numpy as np
import torch
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import roc_auc_score
from apex import amp
from tqdm import trange
import sentencepiece as spm
from fairseq.models.bart import BARTModel
from preprocessor import get_id_text_label_from_csv, get_id_text_from_test_csv

PREDICT = True
USE_PSEUDO = True  # Add pseudo labels to training dataset
PRETRAINED_MODEL = 'models/cc25_pretrain'
PRETRAINED_SPM = 'models/cc25_pretrain/sentence.bpe.model'
TRAIN_SAMPLE_FRAC = 0.05  # what % of training data to use
TRAIN_CSV_PATH = 'data/translated_2018/combined_no_pt.csv'
VAL_CSV_PATH = 'data/validation_en.csv'
PSEUDO_CSV_PATH = 'data/submissions/test9479.csv'
TEST_CSV_PATH = 'data/test.csv'
MAX_CORES = 24  # limit MP calls to use this # cores at most
BASE_MODEL_OUTPUT_DIM = 1024  # hidden layer dimensions
INTERMEDIATE_HIDDEN_UNITS = 1
MAX_SEQ_LEN = 200  # max sequence length for input strings: gets padded/truncated
NUM_EPOCHS = 6  # Half trained using train, half on val (+ PL)
BATCH_SIZE = 16
ACCUM_FOR = 2
LR = 1e-5

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
        self.fc = torch.nn.Linear(BASE_MODEL_OUTPUT_DIM, INTERMEDIATE_HIDDEN_UNITS)

    def forward(self, x):
        hidden_states = self.base_model.extract_features(x)
        logits = self.fc(hidden_states[:, -1, :])
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

            with amp.scale_loss(loss, opt) as scaled_loss:
                scaled_loss.backward()

            running_total_loss += loss.detach().cpu().numpy()
            t.set_postfix(loss=running_total_loss / iter)

            if iter % ACCUM_FOR == 0:
                opt.step()
                opt.zero_grad()


def evaluate(model, val_tuple):
    # Evaluate validation AUC
    val_features, val_labels, val_ids = val_tuple

    model.eval()
    val_preds = []
    with torch.no_grad():
        for batch_idx_start in range(0, len(val_ids), BATCH_SIZE):
            batch_idx_end = min(batch_idx_start + BATCH_SIZE, len(val_ids))
            batch_features = torch.tensor(val_features[batch_idx_start:batch_idx_end]).long().cuda()
            batch_preds = model(batch_features)
            val_preds.append(batch_preds.cpu())

        val_preds = np.concatenate(val_preds)
        val_roc_auc_score = roc_auc_score(val_labels, val_preds)
    return val_roc_auc_score


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
        .to_csv('data/outputs/{}/mbart_{}.csv'.format(save_folder,
                                                      epoch),
                index=False)

    return val_score


def main_driver(train_tuple, val_tuple, test_tuple):
    pretrained_base = BARTModel.from_pretrained(PRETRAINED_MODEL,
                                                checkpoint_file='model.pt',
                                                layernorm_embedding=True)
    classifier = ClassifierHead(pretrained_base).cuda()

    loss_fn = torch.nn.BCELoss()
    opt = torch.optim.Adam(classifier.parameters(), lr=LR)

    amp.register_float_function(torch, 'sigmoid')
    classifier, opt = amp.initialize(classifier, opt, opt_level='O1', verbosity=0)

    list_auc = []

    current_tuple = train_tuple
    for curr_epoch in range(NUM_EPOCHS):
        # Switch from toxic-2018 to the current mixed language dataset, halfway thru
        if curr_epoch == NUM_EPOCHS // 2:
            current_tuple = val_tuple

        train(classifier, current_tuple, loss_fn, opt, curr_epoch)

        epoch_auc = predict_evaluate(classifier, val_tuple, curr_epoch, score=True)
        print('Epoch {} - Raw: {:.4f}'.format(curr_epoch, epoch_auc))
        list_auc.append(epoch_auc)

        if PREDICT:
            predict_evaluate(classifier, test_tuple, curr_epoch)

    with np.printoptions(precision=4, suppress=True):
        print(np.array(list_auc))


if __name__ == '__main__':
    def cln(x):
        return ' '.join(x.split())


    start_time = time.time()

    train_ids, train_strings, train_labels = get_id_text_label_from_csv(TRAIN_CSV_PATH,
                                                                        text_col='comment_text',
                                                                        sample_frac=TRAIN_SAMPLE_FRAC)
    train_strings = [cln(x) for x in train_strings]

    val_ids, val_strings, val_labels = get_id_text_label_from_csv(VAL_CSV_PATH, text_col='comment_text')
    val_strings = [cln(x) for x in val_strings]

    test_ids, test_strings = get_id_text_from_test_csv(TEST_CSV_PATH, text_col='content')
    test_strings = [cln(x) for x in test_strings]

    pseudo_ids, pseudo_labels = [], None
    if USE_PSEUDO:
        pseudo_ids, pseudo_strings, pseudo_labels = get_id_text_label_from_csv(PSEUDO_CSV_PATH, text_col='content')

    tokenizer = spm.SentencePieceProcessor()
    tokenizer.Load(PRETRAINED_SPM)

    print(train_strings[0])
    print(tokenizer.EncodeAsPieces(train_strings[0]))

    print('Encoding raw strings into model-specific tokens')
    train_features = pad_sequences(list(map(tokenizer.EncodeAsIds, train_strings)),
                                   padding='pre',
                                   maxlen=MAX_SEQ_LEN)
    val_features = pad_sequences(list(map(tokenizer.EncodeAsIds, val_strings)),
                                 padding='pre',
                                 maxlen=MAX_SEQ_LEN)
    test_features = pad_sequences(list(map(tokenizer.EncodeAsIds, test_strings)),
                                  padding='post',
                                  maxlen=MAX_SEQ_LEN)

    pseudo_features = None
    if USE_PSEUDO:
        pseudo_features = pad_sequences(list(map(tokenizer.EncodeAsIds, pseudo_strings)),
                                        padding='pre',
                                        maxlen=MAX_SEQ_LEN)

    if USE_PSEUDO:
        train_features = np.concatenate([train_features, pseudo_features])
        train_labels = np.concatenate([train_labels, pseudo_labels])
        train_ids = np.concatenate([train_ids, pseudo_ids])

    print('Train size: {}, val size: {}, pseudo size: {}'.format(len(train_ids), len(val_ids), len(pseudo_ids)))

    main_driver([train_features, train_labels, train_ids],
                [val_features, val_labels, val_ids],
                [test_features, test_ids])

    print('Elapsed time: {}'.format(time.time() - start_time))
