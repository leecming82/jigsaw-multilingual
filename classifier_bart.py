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
from preprocessor import get_id_text_label_from_csv
from torch_helpers import EMA, layerwise_lr_decay

SAVE_MODEL = False
USE_AMP = True
USE_EMA = False
USE_VAL = False  # Train w/ base + validation datasets, turns off scoring
USE_PSEUDO = False  # Add pseudo labels to training dataset
USE_MULTI_GPU = False
USE_LR_DECAY = False
PRETRAINED_MODEL = 'models/cc25_pretrain'
PRETRAINED_SPM = 'models/cc25_pretrain/sentence.bpe.model'
TRAIN_SAMPLE_FRAC = 1  # what % of training data to use
TRAIN_CSV_PATH = 'data/toxic_2018/pl_en.csv'
VAL_CSV_PATH = 'data/validation_en.csv'
PSEUDO_CSV_PATH = 'data/test9383_highconfidence.csv'
OUTPUT_DIR = 'models/high_conf_9383'
NUM_GPUS = 2  # Set to 1 if using AMP (doesn't seem to play nice with 1080 Ti)
MAX_CORES = 24  # limit MP calls to use this # cores at most
BASE_MODEL_OUTPUT_DIM = 1024  # hidden layer dimensions
INTERMEDIATE_HIDDEN_UNITS = 1
MAX_SEQ_LEN = 200  # max sequence length for input strings: gets padded/truncated
NUM_EPOCHS = 6  # Half trained using train, half on val (+ PL)
BATCH_SIZE = 16
ACCUM_FOR = 2
SAVE_ITER = 100  # save every X iterations
EMA_DECAY = 0.999
LR_DECAY_FACTOR = 0.75
LR_DECAY_START = 1e-3
LR_FINETUNE = 1e-5

if not USE_MULTI_GPU:
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
        self.fc = torch.nn.Linear(BASE_MODEL_OUTPUT_DIM, INTERMEDIATE_HIDDEN_UNITS)

    def forward(self, x, freeze=True):
        if freeze:
            with torch.no_grad():
                hidden_states = self.base_model.extract_features(x)
        else:
            hidden_states = self.base_model.extract_features(x)

        logits = self.fc(hidden_states[:, -1, :])
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

            if curr_epoch < 1:
                preds = model(batch_features, freeze=True)
            else:
                preds = model(batch_features, freeze=False)
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


def main_driver(train_tuple, val_raw_tuple, val_translated_tuple, pseudo_tuple):
    pretrained_base = BARTModel.from_pretrained(PRETRAINED_MODEL,
                                                checkpoint_file='model.pt',
                                                layernorm_embedding=True)
    classifier = ClassifierHead(pretrained_base).cuda()

    if USE_EMA:
        ema = EMA(EMA_DECAY)
        for name, param in classifier.named_parameters():
            if param.requires_grad:
                ema.register(name, param.data)
    else:
        ema = None

    loss_fn = torch.nn.BCELoss()
    if USE_LR_DECAY:
        parameters_update = layerwise_lr_decay(classifier, LR_DECAY_START, LR_DECAY_FACTOR)
        opt = torch.optim.Adam(parameters_update)
    else:
        opt = torch.optim.Adam(classifier.parameters(), lr=LR_DECAY_START)

    if USE_AMP:
        amp.register_float_function(torch, 'sigmoid')
        classifier, opt = amp.initialize(classifier, opt, opt_level='O1', verbosity=0)

    if USE_MULTI_GPU:
        classifier = torch.nn.DataParallel(classifier)

    list_raw_auc, list_translated_auc = [], []

    current_tuple = train_tuple
    for curr_epoch in range(NUM_EPOCHS):
        # switch to finetune - only lower for those > finetune LR
        # i.e., lower layers might have even smaller LR
        if curr_epoch == 1:
            print('Switching to fine-tune LR')
            for g in opt.param_groups:
                g['lr'] = LR_FINETUNE

        # Switch from toxic-2018 to the current mixed language dataset, halfway thru
        if curr_epoch == NUM_EPOCHS // 2:
            if USE_PSEUDO:
                current_tuple = pseudo_tuple
            else:
                current_tuple = val_raw_tuple

        train(classifier, current_tuple, loss_fn, opt, curr_epoch, ema)

        epoch_raw_auc = evaluate(classifier, val_raw_tuple)
        epoch_translated_auc = evaluate(classifier, val_translated_tuple)
        print('Epoch {} - Raw: {:.4f}, Translated: {:.4f}'.format(curr_epoch, epoch_raw_auc, epoch_translated_auc))
        list_raw_auc.append(epoch_raw_auc)
        list_translated_auc.append(epoch_translated_auc)

    with np.printoptions(precision=4, suppress=True):
        print(np.array(list_raw_auc))
        print(np.array(list_translated_auc))


if __name__ == '__main__':
    start_time = time.time()

    train_ids, train_strings, train_labels = get_id_text_label_from_csv(TRAIN_CSV_PATH,
                                                                        text_col='comment_text',
                                                                        sample_frac=TRAIN_SAMPLE_FRAC)
    val_ids, val_raw_strings, val_labels = get_id_text_label_from_csv(VAL_CSV_PATH, text_col='comment_text')
    _, val_translated_strings, _ = get_id_text_label_from_csv(VAL_CSV_PATH, text_col='comment_text_en')

    pseudo_ids, pseudo_labels = None, None
    if USE_PSEUDO:
        pseudo_ids, pseudo_strings, pseudo_labels = get_id_text_label_from_csv(PSEUDO_CSV_PATH, text_col='content')

    tokenizer = spm.SentencePieceProcessor()
    tokenizer.Load(PRETRAINED_SPM)

    print('Encoding raw strings into model-specific tokens')
    train_features = pad_sequences(list(map(tokenizer.EncodeAsIds, train_strings)),
                                   padding='post',
                                   maxlen=MAX_SEQ_LEN)
    val_raw_features = pad_sequences(list(map(tokenizer.EncodeAsIds, val_raw_strings)),
                                     padding='post',
                                     maxlen=MAX_SEQ_LEN)
    val_translated_features = pad_sequences(list(map(tokenizer.EncodeAsIds, val_translated_strings)),
                                            padding='post',
                                            maxlen=MAX_SEQ_LEN)
    pseudo_features = None
    if USE_PSEUDO:
        pseudo_features = pad_sequences(list(map(tokenizer.EncodeAsIds, pseudo_strings)),
                                        padding='post',
                                        maxlen=MAX_SEQ_LEN)

    print(train_features.shape, val_raw_features.shape)
    print(train_features[0, :])

    if USE_VAL:
        train_features = np.concatenate([train_features, val_raw_features])
        train_labels = np.concatenate([train_labels, val_labels])
        train_ids = np.concatenate([train_ids, val_ids])

    if USE_PSEUDO:  # so that when we switch, can just use this tuple imeddiately
        pseudo_features = np.concatenate([val_raw_features, pseudo_features])
        pseudo_labels = np.concatenate([val_labels, pseudo_labels])
        pseudo_ids = np.concatenate([val_ids, pseudo_ids])

    main_driver([train_features, train_labels, train_ids],
                [val_raw_features, val_labels, val_ids],
                [val_translated_features, val_labels, val_ids],
                [pseudo_features, pseudo_labels, pseudo_ids])

    print('Elapsed time: {}'.format(time.time() - start_time))
