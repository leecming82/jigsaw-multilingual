"""
ELMo classifier built off AllenAI NLP torch implementation
"""
import os
import time
from random import shuffle
from functools import partial
import multiprocessing as mp
import pandas as pd
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from apex import amp
from tqdm import trange
from preprocessor import get_id_text_label_from_csv, get_id_text_from_test_csv
from allennlp.modules.elmo import Elmo, batch_to_ids
from allennlp.modules.seq2vec_encoders import PytorchSeq2VecWrapper

RUN_NAME = '9544ru_elmo'  # added as prefix to file outputs
PREDICT = True  # Make predictions against TEST_CSV_PATH test features
SAVE_MODEL = False  # Saves model at end of every epoch to MODEL_SAVE_DIR
USE_VAL_LANG = 'ru'  # if set to ISO lang str (e.g., "es") - only pulls that language's validation samples
PRETRAINED_MODEL_WEIGHTS = 'models/russian-taiga/model.hdf5'
PRETRAINED_MODEL_OPTIONS = 'models/russian-taiga/options.json'
TRAIN_SAMPLE_FRAC = 1.  # what proportion of training data (from TRAIN_CSV_PATH) to sample
TRAIN_CSV_PATH = 'data/ru_all.csv'
TEST_CSV_PATH = 'data/ru_test.csv'
VAL_CSV_PATH = 'data/validation.csv'
MODEL_SAVE_DIR = 'models/{}'.format(RUN_NAME)
MAX_CORES = 24  # limit MP calls to use this # cores at most; for tokenizing
BASE_MODEL_OUTPUT_DIM = 2048  # hidden layer dimensions
NUM_OUTPUTS = 1  # Num of output units (should be 1 for Toxicity)
MAX_SEQ_LEN = 200  # max sequence length for input strings: gets padded/truncated
NUM_EPOCHS = 4
# Gradient Accumulation: updates every ACCUM_FOR steps so that effective BS = BATCH_SIZE * ACCUM_FOR
BATCH_SIZE = 64
ACCUM_FOR = 1
LR = 1e-3  # Learning rate - constant value

# For multi-gpu environments - make only 1 GPU visible to process
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'


class ClassifierHead(torch.nn.Module):
    def __init__(self):
        super(ClassifierHead, self).__init__()
        self.base_model = Elmo(PRETRAINED_MODEL_OPTIONS, PRETRAINED_MODEL_WEIGHTS, 1, dropout=0)

        self.conv1 = torch.nn.Conv1d(1024, 16, 3)
        self.p1 = torch.nn.AdaptiveMaxPool1d(128)
        self.activation_func = torch.nn.ReLU6()
        self.hidden2label = torch.nn.Linear(2048, NUM_OUTPUTS)
        # self.encoder = PytorchSeq2VecWrapper(torch.nn.LSTM(1024,
        #                                                    128,
        #                                                    2,
        #                                                    bidirectional=True,
        #                                                    batch_first=True))
        # self.fc = torch.nn.Linear(128 * 2, 1)

    def forward(self, x):
        with torch.no_grad():
            raw_elmo_output = self.base_model(x)['elmo_representations']
        hidden_states = torch.cat(raw_elmo_output, -1)

        x = hidden_states.transpose(1, 2)
        x = self.conv1(x)
        x = self.activation_func(x)
        x = self.p1(x)
        x = x.view(-1, 2048)
        prob = torch.nn.Sigmoid()(self.hidden2label(x))

        # x = self.encoder(hidden_states, mask=None)
        # x = self.fc(x)
        # prob = torch.nn.Sigmoid()(x)
        return prob


def train(model, train_tuple, loss_fn, opt, curr_epoch):
    """
    Trains against the train_tuple features for a single epoch
    """
    # Shuffle train indices for current epoch, batching
    all_features, all_labels, all_ids = train_tuple
    train_indices = list(range(len(all_labels)))

    shuffle(train_indices)
    train_features = all_features[train_indices]
    train_labels = all_labels[train_indices]

    model.train()
    iter = 0
    running_total_loss = 0  # Display running average of loss across epoch
    with trange(0, len(train_indices), BATCH_SIZE,
                desc='Epoch {}'.format(curr_epoch)) as t:
        for batch_idx_start in t:
            iter += 1
            batch_idx_end = min(batch_idx_start + BATCH_SIZE, len(train_indices))
            batch_features = batch_to_ids(train_features[batch_idx_start:batch_idx_end]).cuda()
            batch_labels = torch.tensor(train_labels[batch_idx_start:batch_idx_end]).float().cuda().unsqueeze(-1)
            preds = model(batch_features)
            loss = loss_fn(preds, batch_labels)
            loss = loss / ACCUM_FOR  # Normalize if we're doing GA

            with amp.scale_loss(loss, opt) as scaled_loss:
                scaled_loss.backward()

            running_total_loss += loss.detach().cpu().numpy()
            t.set_postfix(loss=running_total_loss / iter)

            if iter % ACCUM_FOR == 0:
                opt.step()
                opt.zero_grad()


def predict_evaluate(model, data_tuple, epoch, score=False):
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
            batch_features = batch_to_ids(data_tuple[0][batch_idx_start:batch_idx_end]).cuda()
            batch_preds = model(batch_features)
            val_preds.append(batch_preds.cpu().numpy().squeeze())

    val_preds = np.concatenate(val_preds)
    if score:
        val_score = roc_auc_score(np.round(data_tuple[1]), val_preds)

    save_folder = 'validation' if score else 'test'
    pd.DataFrame({'id': data_tuple[-1], 'toxic': val_preds}) \
        .to_csv('data/outputs/{}/{}_{}.csv'.format(save_folder,
                                                   RUN_NAME,
                                                   epoch),
                index=False)

    return val_score


def main_driver(train_tuple, val_tuple, test_tuple):
    classifier = ClassifierHead().cuda()
    loss_fn = torch.nn.BCELoss()
    opt = torch.optim.Adam(classifier.parameters(), lr=LR)

    amp.register_float_function(torch, 'sigmoid')
    classifier, opt = amp.initialize(classifier, opt, opt_level='O1', verbosity=0)
    list_auc = []

    current_tuple = train_tuple
    for curr_epoch in range(NUM_EPOCHS):
        # After half epochs, switch to training against validation set
        # if curr_epoch == NUM_EPOCHS // 2:
        #     current_tuple = val_tuple
        train(classifier, current_tuple, loss_fn, opt, curr_epoch)

        # epoch_raw_auc = predict_evaluate(classifier, val_tuple, curr_epoch, score=True)
        # print('Epoch {} - Raw: {:.4f}'.format(curr_epoch, epoch_raw_auc))
        # list_auc.append(epoch_raw_auc)

        if PREDICT:
            predict_evaluate(classifier, test_tuple, curr_epoch)

    with np.printoptions(precision=4, suppress=True):
        print(np.array(list_auc))

    pd.DataFrame({'val_auc': list_auc}).to_csv('data/outputs/results/{}.csv'.format(RUN_NAME), index=False)


if __name__ == '__main__':
    def cln(x):  # Truncates adjacent whitespaces to single whitespace
        return x.split()[:MAX_SEQ_LEN]  # manage max seq length at the word token level


    start_time = time.time()
    print(RUN_NAME)

    # Load train, validation, and pseudo-label data
    train_ids, train_strings, train_labels = get_id_text_label_from_csv(TRAIN_CSV_PATH,
                                                                        text_col='comment_text',
                                                                        sample_frac=TRAIN_SAMPLE_FRAC)
    print(train_strings[0])
    train_strings = np.array([cln(x) for x in train_strings])
    print(train_strings[0])

    val_ids, val_strings, val_labels = get_id_text_label_from_csv(VAL_CSV_PATH,
                                                                  text_col='comment_text',
                                                                  lang=USE_VAL_LANG)
    val_strings = np.array([cln(x) for x in val_strings])

    test_ids, test_strings = get_id_text_from_test_csv(TEST_CSV_PATH, text_col='comment_text')
    test_strings = np.array([cln(x) for x in test_strings])

    print('Train size: {}, val size: {}'.format(len(train_ids), len(val_ids)))
    print('Train positives: {}, train negatives: {}'.format(train_labels[train_labels == 1].shape,
                                                            train_labels[train_labels == 0].shape))

    main_driver([train_strings, train_labels, train_ids],
                [val_strings, val_labels, val_ids],
                [test_strings, test_ids])

    print('Elapsed time: {}'.format(time.time() - start_time))
