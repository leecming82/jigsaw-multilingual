"""
Modification of baseline classifier w/ additional unsupervised consistency loss task (UDA)
"""
import os
import time
from itertools import starmap
from random import shuffle
from functools import partial
import multiprocessing as mp
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel, AutoConfig
from sklearn.metrics import roc_auc_score
from apex import amp
from tqdm import trange
from preprocessor import get_id_text_label_from_csv, get_id_text_from_test_csv
from torch_helpers import EMA, save_model, layerwise_lr_decay
from classifier_baseline import evaluate, ClassifierHead
from classifier_baseline import train as pretrain

SAVE_MODEL = True
USE_AMP = True
USE_EMA = False
USE_MULTI_GPU = False
USE_LR_DECAY = False
PRETRAINED_MODEL = 'xlm-roberta-large'
TRAIN_SAMPLE_FRAC = 1  # what % of training data to use
TRAIN_CSV_PATH = 'data/toxic_2018/pl_en.csv'
VAL_CSV_PATH = 'data/validation_en.csv'
UNSUP_RAW_CSV_PATH = 'data/test_en.csv'
UNSUP_AUG_CSV_PATH = 'data/test_en.csv'
OUTPUT_DIR = 'models/UDA'
NUM_GPUS = 2  # Set to 1 if using AMP (doesn't seem to play nice with 1080 Ti)
MAX_CORES = 24  # limit MP calls to use this # cores at most
BASE_MODEL_OUTPUT_DIM = 1024  # hidden layer dimensions
INTERMEDIATE_HIDDEN_UNITS = 1
MAX_SEQ_LEN = 200  # max sequence length for input strings: gets padded/truncated
NUM_EPOCHS = 4  # Half trained using train, half on val (+ PL)
SUP_BATCH_SIZE = 8  # Supervised batch size
UNSUP_BATCH_SIZE = 16  # Unsupervised batch size
ACCUM_FOR = 2
SAVE_ITER = 100  # save every X iterations
EMA_DECAY = 0.999
LR_DECAY_FACTOR = 0.75
LR_DECAY_START = 1e-3
LR_FINETUNE = 1e-5

if not USE_MULTI_GPU:
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'


def train(model, config,
          train_tuple, supervised_loss_fn,
          unsupervised_tuple, unsupervised_loss_fn,
          opt,
          curr_epoch,
          ema):
    """ Train """
    # Shuffle supervised samples for current epoch, batching
    sup_features, sup_labels, _ = train_tuple
    sup_indices = list(range(len(sup_labels)))
    shuffle(sup_indices)
    sup_features, sup_labels = sup_features[sup_indices], sup_labels[sup_indices]

    # Shuffle unsupervised samples
    unsup_raw_features, unsup_aug_features = unsupervised_tuple
    unsup_indices = list(range(len(unsup_raw_features)))
    shuffle(unsup_indices)
    unsup_raw_features, unsup_aug_features = unsup_raw_features[unsup_indices], unsup_aug_features[unsup_indices]

    model.train()
    iter = 0
    running_supervised_loss = 0
    running_unsupervised_loss = 0
    with trange(0, len(sup_indices) // SUP_BATCH_SIZE, desc='{}'.format(curr_epoch)) as t:
        for curr_batch_index in t:
            iter += 1

            # Calculated supervised loss
            batch_sup_start_idx = SUP_BATCH_SIZE * curr_batch_index
            batch_sup_idx_end = min(batch_sup_start_idx + SUP_BATCH_SIZE, len(sup_labels))
            batch_sup_features = torch.tensor(sup_features[batch_sup_start_idx:batch_sup_idx_end]).cuda()
            batch_sup_labels = torch.tensor(sup_labels[batch_sup_start_idx:batch_sup_idx_end]).float().cuda().unsqueeze(
                -1)
            batch_sup_preds = model(batch_sup_features, freeze=False)
            supervised_loss = supervised_loss_fn(batch_sup_preds, batch_sup_labels)

            # Calculate unsupervised loss
            batch_unsup_start_idx = UNSUP_BATCH_SIZE * curr_batch_index
            batch_unsup_idx_end = min(batch_unsup_start_idx + UNSUP_BATCH_SIZE, len(unsup_raw_features))
            batch_unsup_raw_features = torch.tensor(
                unsup_raw_features[batch_unsup_start_idx:batch_unsup_idx_end]).cuda()
            batch_unsup_aug_features = torch.tensor(
                unsup_aug_features[batch_unsup_start_idx:batch_unsup_idx_end]).cuda()
            with torch.no_grad():
                batch_unsup_raw_preds = model(batch_unsup_raw_features, freeze=False)
                scaled_positive = torch.pow(batch_unsup_raw_preds, 1. / 0.5)
                denominator = (scaled_positive + torch.pow(1 - batch_unsup_raw_preds, 1. / 0.5))
                batch_unsup_raw_preds = scaled_positive / denominator
            batch_unsup_aug_preds = model(batch_unsup_aug_features, freeze=False)
            unsupervised_loss = unsupervised_loss_fn(batch_unsup_aug_preds, batch_unsup_raw_preds)

            combined_loss = (supervised_loss + unsupervised_loss).mean()

            if USE_AMP:
                with amp.scale_loss(combined_loss, opt) as scaled_loss:
                    scaled_loss.backward()
            else:
                combined_loss.backward()

            running_supervised_loss += supervised_loss.detach().cpu().numpy()
            running_unsupervised_loss += unsupervised_loss.detach().cpu().numpy()
            t.set_postfix(sup_loss=running_supervised_loss / iter, unsup_loss=running_unsupervised_loss / iter)

            if iter % ACCUM_FOR == 0:
                opt.step()
                opt.zero_grad()

            if USE_EMA:
                # Update EMA shadow parameters on every back pass
                for name, param in model.named_parameters():
                    if param.requires_grad:
                        ema.update(name, param.data)

            # Save every specified step on last epoch
            if curr_epoch == NUM_EPOCHS - 1 and iter % SAVE_ITER == 0 and SAVE_MODEL:
                print('Saving epoch {} model'.format(curr_epoch))
                save_model(os.path.join(OUTPUT_DIR, PRETRAINED_MODEL, '{}_{}'.format(curr_epoch, iter)),
                           model,
                           config,
                           tokenizer)


def main_driver(train_tuple, val_raw_tuple, val_translated_tuple, unsupervised_tuple, tokenizer):
    pretrained_config = AutoConfig.from_pretrained(PRETRAINED_MODEL,
                                                   output_hidden_states=True)
    pretrained_base = AutoModel.from_pretrained(PRETRAINED_MODEL, config=pretrained_config).cuda()
    classifier = ClassifierHead(pretrained_base).cuda()

    if USE_EMA:
        ema = EMA(EMA_DECAY)
        for name, param in classifier.named_parameters():
            if param.requires_grad:
                ema.register(name, param.data)
    else:
        ema = None

    supervised_loss_fn = torch.nn.BCELoss()
    unsupervised_loss_fn = torch.nn.BCELoss()

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

    for curr_epoch in range(NUM_EPOCHS):
        # switch to finetune - only lower for those > finetune LR
        # i.e., lower layers might have even smaller LR
        if curr_epoch == 1:
            print('Switching to fine-tune LR')
            for g in opt.param_groups:
                g['lr'] = LR_FINETUNE

        # Switch from toxic-2018 to the current mixed language dataset, halfway thru
        if curr_epoch < NUM_EPOCHS // 2:
            pretrain(classifier,
                     pretrained_config,
                     train_tuple,
                     supervised_loss_fn,
                     opt,
                     curr_epoch,
                     ema)
        else:
            train(classifier,
                  pretrained_config,
                  val_raw_tuple, supervised_loss_fn,
                  unsupervised_tuple, unsupervised_loss_fn,
                  opt,
                  curr_epoch,
                  ema)

        epoch_raw_auc = evaluate(classifier, val_raw_tuple)
        epoch_translated_auc = evaluate(classifier, val_translated_tuple)
        print('Epoch {} - Raw: {:.4f}, Translated: {:.4f}'.format(curr_epoch, epoch_raw_auc, epoch_translated_auc))
        list_raw_auc.append(epoch_raw_auc)
        list_translated_auc.append(epoch_translated_auc)

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
        save_model(os.path.join(OUTPUT_DIR, '{}_ema'.format(PRETRAINED_MODEL)), classifier, pretrained_config,
                   tokenizer)


if __name__ == '__main__':
    start_time = time.time()

    # Load train, validation, and pseudo-label data
    train_ids, train_strings, train_labels = get_id_text_label_from_csv(TRAIN_CSV_PATH,
                                                                        text_col='comment_text',
                                                                        sample_frac=TRAIN_SAMPLE_FRAC)

    val_ids, val_raw_strings, val_labels = get_id_text_label_from_csv(VAL_CSV_PATH, text_col='comment_text')
    _, val_translated_strings, _ = get_id_text_label_from_csv(VAL_CSV_PATH, text_col='comment_text_en')

    unsup_ids, unsup_raw_strings = get_id_text_from_test_csv(UNSUP_RAW_CSV_PATH, text_col='content')
    unsup_aug_ids, unsup_aug_strings = get_id_text_from_test_csv(UNSUP_AUG_CSV_PATH, text_col='content_en')
    assert (unsup_ids == unsup_aug_ids).all()
    assert len(unsup_raw_strings) == len(unsup_aug_strings)

    # use MP to batch encode the raw feature strings into Bert token IDs
    tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_MODEL)
    if 'gpt' in PRETRAINED_MODEL:  # GPT2 pre-trained tokenizer doesn't set a padding token
        tokenizer.add_special_tokens({'pad_token': '<|endoftext|>'})

    encode_partial = partial(tokenizer.encode,
                             max_length=MAX_SEQ_LEN,
                             pad_to_max_length=True,
                             add_special_tokens=True)
    print('Encoding raw strings into model-specific tokens')
    with mp.Pool(MAX_CORES) as p:
        train_features = np.array(p.map(encode_partial, train_strings))
        val_raw_features = np.array(p.map(encode_partial, val_raw_strings))
        val_translated_features = np.array(p.map(encode_partial, val_translated_strings))
        unsup_raw_features = np.array(p.map(encode_partial, unsup_raw_strings))
        unsup_aug_features = np.array(p.map(encode_partial, unsup_aug_strings))

    print('Train size: {}, val size: {}, unsupervised size: {}'.format(len(train_ids), len(val_ids),
                                                                       len(unsup_raw_features)))

    main_driver([train_features, train_labels, train_ids],
                [val_raw_features, val_labels, val_ids],
                [val_translated_features, val_labels, val_ids],
                [unsup_raw_features, unsup_aug_features],
                tokenizer)

    print('Elapsed time: {}'.format(time.time() - start_time))
