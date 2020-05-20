"""
Torch classifier:
- Flair embeddings combined with Bidirectional GRU
"""
import os
import time
from random import shuffle
import pandas as pd
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from flair.data import Sentence
from flair.embeddings import FlairEmbeddings, StackedEmbeddings, WordEmbeddings, FastTextEmbeddings
from sklearn.metrics import roc_auc_score
from apex import amp
from tqdm import trange
from preprocessor import get_id_text_label_from_csv, get_id_text_from_test_csv

RUN_NAME = '9544es_flair'  # added as prefix to file outputs
PREDICT = True  # Make predictions against TEST_CSV_PATH test features
USE_PSEUDO = False  # Use pseudo-labels in PSEUDO_CSV_PATH
SAVE_MODEL = False  # Saves model at end of every epoch to MODEL_SAVE_DIR
USE_VAL_LANG = 'es'  # if set to ISO lang str (e.g., "es") - only pulls that language's validation samples
TRAIN_SAMPLE_FRAC = 1.  # what proportion of training data (from TRAIN_CSV_PATH) to sample
TRAIN_CSV_PATH = 'data/es_all.csv'
TEST_CSV_PATH = 'data/es_test.csv'
PSEUDO_CSV_PATH = 'data/submissions/test9529.csv'
VAL_CSV_PATH = 'data/validation.csv'
FLAIR_EMBEDDING_DIM = 300
FLAIR_MODEL_LIST = [FastTextEmbeddings('models/cc.es.300.bin')]
# FLAIR_MODEL_LIST = [FlairEmbeddings('multi-v0-forward-fast'),
#                     FlairEmbeddings('multi-v0-backward-fast')]
RNN_LAYERS = 2
RNN_HIDDEN = 128
NUM_OUTPUTS = 1  # Num of output units (should be 1 for Toxicity)
MAX_SEQ_LEN = 200  # max sequence length for input strings: gets padded/truncated
NUM_EPOCHS = 4
# Gradient Accumulation: updates every ACCUM_FOR steps so that effective BS = BATCH_SIZE * ACCUM_FOR
BATCH_SIZE = 64
ACCUM_FOR = 1
LR = 1e-3  # Learning rate - constant value

# For multi-gpu environments - make only 1 GPU visible to process
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


class BiRNN(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(BiRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = torch.nn.GRU(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.fc = torch.nn.Linear(hidden_size * 2, num_classes)  # 2 for bidirection

    def forward(self, x):
        out, _ = self.gru(x)  # out: tensor of shape (batch_size, seq_length, hidden_size*2)

        # Decode the hidden state of the last time step
        out = torch.nn.Sigmoid()(self.fc(out[:, -1, :]))
        return out


def train(stacked_embeddings, model, train_tuple, loss_fn, opt, curr_epoch):
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

            batch_sentences = [Sentence(x, use_tokenizer=True) for x in train_features[batch_idx_start:batch_idx_end]]
            stacked_embeddings.embed(batch_sentences)
            batch_features = [torch.stack([curr_token.embedding for curr_token in curr_sentence]) for curr_sentence
                              in batch_sentences]
            batch_features = pad_sequence(batch_features, batch_first=True)[:, :MAX_SEQ_LEN, :]
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


def predict_evaluate(stacked_embeddings, model, data_tuple, epoch, score=False):
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
            batch_sentences = [Sentence(x, use_tokenizer=True) for x in data_tuple[0][batch_idx_start:batch_idx_end]]
            stacked_embeddings.embed(batch_sentences)
            batch_features = [torch.stack([curr_token.embedding for curr_token in curr_sentence]) for curr_sentence
                              in batch_sentences]
            batch_features = pad_sequence(batch_features, batch_first=True)[:, :MAX_SEQ_LEN, :]
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
    stacked_embeddings = StackedEmbeddings(embeddings=FLAIR_MODEL_LIST)
    classifier = BiRNN(FLAIR_EMBEDDING_DIM, RNN_HIDDEN, RNN_LAYERS, NUM_OUTPUTS).cuda()
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
        train(stacked_embeddings, classifier, current_tuple, loss_fn, opt, curr_epoch)

        epoch_raw_auc = predict_evaluate(stacked_embeddings, classifier, val_tuple, curr_epoch, score=True)
        print('Epoch {} - Raw: {:.4f}'.format(curr_epoch, epoch_raw_auc))
        list_auc.append(epoch_raw_auc)

        if PREDICT:
            predict_evaluate(stacked_embeddings, classifier, test_tuple, curr_epoch)

    with np.printoptions(precision=4, suppress=True):
        print(np.array(list_auc))

    pd.DataFrame({'val_auc': list_auc}).to_csv('data/outputs/results/{}.csv'.format(RUN_NAME), index=False)


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
