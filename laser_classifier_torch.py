"""
Use pretrained LASER embeddings w/ an MLP to classify
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
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel, AutoConfig, WEIGHTS_NAME, CONFIG_NAME
from sklearn.metrics import roc_auc_score
from apex import amp
from tqdm import trange
from preprocessor import get_id_text_label_from_csv, get_id_text_distill_label_from_csv
from torch_helpers import EMA, save_model

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['LASER'] = '/root/LASER'
from LASER.source.embed import SentenceEncoder

SAVE_MODEL = False
USE_AMP = True
USE_EMA = False
USE_DISTILL = True
PRETRAINED_MODEL = 'distilbert-base-uncased'
LASER_MODEL_PATH = '/root/LASER/models/bilstm.93langs.2018-12-26.pt'
# TRAIN_CSV_PATH = 'data/jigsaw-toxic-comment-train.csv'
# DISTIL_CSV_PATH = None
TRAIN_CSV_PATH = 'data/toxic_2018/train.csv'
DISTIL_CSV_PATH = 'data/toxic_2018/ensemble_3.csv'
VAL_CSV_PATH = 'data/validation_en.csv'
OUTPUT_DIR = 'models/'
NUM_GPUS = 2  # Set to 1 if using AMP (doesn't seem to play nice with 1080 Ti)
MAX_CORES = 24  # limit MP calls to use this # cores at most
ENCODER_DIM = 1024
NUM_HIDDEN = 256
OUTPUT_DIM = 1
MAX_SEQ_LEN = 200  # max sequence length for input strings: gets padded/truncated
NUM_EPOCHS = 5
BATCH_SIZE = 64
EMA_DECAY = 0.999


class mlp(torch.nn.Module):
    """
    Bert base with a Linear layer plopped on top of it
    - connects the max pool of the last hidden layer with the FC
    """

    def __init__(self):
        super(mlp, self).__init__()
        self.fc_1 = torch.nn.Linear(ENCODER_DIM, NUM_HIDDEN)
        self.fc_2 = torch.nn.Linear(NUM_HIDDEN, OUTPUT_DIM)

    def forward(self, x):
        hidden = self.fc_1(x)
        hidden = F.tanh(hidden)
        logits = self.fc_2(hidden)
        prob = torch.nn.Sigmoid()(logits)
        return prob


def train(model, train_tuple, loss_fn, opt, curr_epoch, ema):
    """ Train """
    # Shuffle train indices for current epoch, batching
    all_features, all_labels, all_ids = train_tuple
    train_indices = list(range(len(all_labels)))

    shuffle(train_indices)
    train_features = all_features[train_indices]
    train_labels = all_labels[train_indices]

    # # switch to finetune
    # if curr_epoch == 1:
    #     for g in opt.param_groups:
    #         g['lr'] = 1e-5

    model.train()
    with trange(0, len(train_indices), BATCH_SIZE,
                desc='Epoch {}'.format(curr_epoch)) as t:
        for batch_idx_start in t:
            opt.zero_grad()
            batch_idx_end = min(batch_idx_start + BATCH_SIZE, len(train_indices))

            batch_features = torch.tensor(train_features[batch_idx_start:batch_idx_end]).cuda()
            batch_labels = torch.tensor(train_labels[batch_idx_start:batch_idx_end]).float().cuda().unsqueeze(-1)

            preds = model(batch_features)
            loss = loss_fn(preds, batch_labels)

            if USE_AMP:
                with amp.scale_loss(loss, opt) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            opt.step()

            if USE_EMA:
                # Update EMA shadow parameters on every back pass
                for name, param in model.named_parameters():
                    if param.requires_grad:
                        ema.update(name, param.data)


def evaluate(model, val_tuple):
    # Evaluate validation AUC
    val_features, val_labels, val_ids = val_tuple

    model.eval()
    val_preds = []
    with torch.no_grad():
        for batch_idx_start in range(0, len(val_ids), BATCH_SIZE):
            batch_idx_end = min(batch_idx_start + BATCH_SIZE, len(val_ids))
            batch_features = torch.tensor(val_features[batch_idx_start:batch_idx_end]).cuda()
            batch_preds = model(batch_features)
            val_preds.append(batch_preds.cpu())

        val_preds = np.concatenate(val_preds)
        val_roc_auc_score = roc_auc_score(val_labels, val_preds)
    return val_roc_auc_score


def main_driver(train_tuple, val_raw_tuple, val_translated_tuple):
    classifier = mlp().cuda()

    if USE_EMA:
        ema = EMA(EMA_DECAY)
        for name, param in classifier.named_parameters():
            if param.requires_grad:
                ema.register(name, param.data)
    else:
        ema = None

    loss_fn = torch.nn.BCELoss()
    opt = torch.optim.Adam(classifier.parameters(), lr=1e-3)

    if USE_AMP:
        amp.register_float_function(torch, 'sigmoid')
        classifier, opt = amp.initialize(classifier, opt, opt_level='O1', verbosity=0)

    best_raw_auc, best_translated_auc = -1, -1
    list_raw_auc, list_translated_auc = [], []
    for curr_epoch in range(NUM_EPOCHS):
        train(classifier, train_tuple, loss_fn, opt, curr_epoch, ema)

        epoch_raw_auc = evaluate(classifier, val_raw_tuple)
        epoch_translated_auc = evaluate(classifier, val_translated_tuple)
        print('Epoch {} - Raw: {:.4f}, Translated: {:.4f}'.format(curr_epoch, epoch_raw_auc, epoch_translated_auc))
        list_raw_auc.append(epoch_raw_auc)
        list_translated_auc.append(epoch_translated_auc)

        if epoch_raw_auc > best_raw_auc and SAVE_MODEL:
            print('Raw AUC increased; saving model')
            best_raw_auc = epoch_raw_auc
            # save_model(os.path.join(OUTPUT_DIR, PRETRAINED_MODEL), classifier, pretrained_config, tokenizer)
        elif epoch_translated_auc > best_translated_auc and SAVE_MODEL:
            print('Raw AUC increased; saving model')
            best_translated_auc = epoch_translated_auc
            # save_model(os.path.join(OUTPUT_DIR, PRETRAINED_MODEL), classifier, pretrained_config, tokenizer)

    with np.printoptions(precision=4, suppress=True):
        print(np.array(list_raw_auc))
        print(np.array(list_translated_auc))

    if USE_EMA and SAVE_MODEL:
        # Load EMA parameters and evaluate once again
        for name, param in classifier.named_parameters():
            if param.requires_grad:
                param.data = ema.get(name)
        epoch_raw_auc = evaluate(classifier, val_raw_tuple)
        epoch_translated_auc = evaluate(classifier, val_translated_tuple)
        print('EMA - Raw: {:.4f}, Translated: {:.4f}'.format(epoch_raw_auc, epoch_translated_auc))
        # save_model(os.path.join(OUTPUT_DIR, '{}_ema'.format(PRETRAINED_MODEL)), classifier, pretrained_config, tokenizer)


if __name__ == '__main__':
    start_time = time.time()

    if USE_DISTILL:
        train_ids, train_strings, train_labels = get_id_text_distill_label_from_csv(TRAIN_CSV_PATH, DISTIL_CSV_PATH)
    else:
        train_ids, train_strings, train_labels = get_id_text_label_from_csv(TRAIN_CSV_PATH)
    val_ids, val_raw_strings, val_labels = get_id_text_label_from_csv(VAL_CSV_PATH, text_col='comment_text')
    _, val_translated_strings, _ = get_id_text_label_from_csv(VAL_CSV_PATH, text_col='comment_text_en')

    sentence_enc = SentenceEncoder(LASER_MODEL_PATH, max_tokens=200, verbose=True, fp16=True)
    train_features = sentence_enc.encode_sentences(train_strings[:10000])
    val_raw_features = sentence_enc.encode_sentences(val_raw_strings)
    val_translated_features = sentence_enc.encode_sentences(val_translated_strings)
    train_labels = train_labels[:10000]
    val_labels = val_labels
    print(train_features.shape, val_raw_features.shape, val_translated_features.shape)

    main_driver([train_features, train_labels, train_ids],
                [val_raw_features, val_labels, val_ids],
                [val_translated_features, val_labels, val_ids])

    print('ELapsed time: {}'.format(time.time() - start_time))