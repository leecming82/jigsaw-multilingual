"""
Combo torch Transformer + USE classifier
"""
import os
import time
from random import shuffle
from functools import partial
import multiprocessing as mp
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel, AutoConfig, WEIGHTS_NAME, CONFIG_NAME
from sklearn.metrics import roc_auc_score
from apex import amp
from tqdm import trange
from preprocessor import get_id_text_label_from_csv
from torch_helpers import EMA, save_model

SAVE_MODEL = True
USE_AMP = True
USE_EMA = False
USE_MULTI_GPU = False
USE_MODEL_PATH = './models/use_ml/medium'
PRETRAINED_MODEL = 'distilbert-base-uncased'
TRAIN_CSV_PATH = 'data/toxic_2018/pl_en.csv'
VAL_CSV_PATH = 'data/validation_en.csv'
OUTPUT_DIR = 'models/'
NUM_GPUS = 2  # Set to 1 if using AMP (doesn't seem to play nice with 1080 Ti)
MAX_CORES = 24  # limit MP calls to use this # cores at most
BASE_MODEL_OUTPUT_DIM = 768  # hidden layer dimensions
INTERMEDIATE_HIDDEN_UNITS = 1
MAX_SEQ_LEN = 200  # max sequence length for input strings: gets padded/truncated
NUM_EPOCHS = 5
BATCH_SIZE = 64
EMA_DECAY = 0.999

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
        self.cnn = torch.nn.Conv1d(BASE_MODEL_OUTPUT_DIM, INTERMEDIATE_HIDDEN_UNITS, kernel_size=1)
        self.fc = torch.nn.Linear(768 + 512, INTERMEDIATE_HIDDEN_UNITS)

    def forward(self, x, y, freeze=True):
        if freeze:
            with torch.no_grad():
                hidden_states = self.base_model(x)[0]
        else:
            hidden_states = self.base_model(x)[0]

        # hidden_states = hidden_states.permute(0, 2, 1)
        # cnn_states = self.cnn(hidden_states)
        # cnn_states = cnn_states.permute(0, 2, 1)
        # logits, _ = torch.max(cnn_states, 1)

        hidden_states = torch.cat([hidden_states[:, 0, :], y], -1)
        logits = self.fc(hidden_states)
        prob = torch.nn.Sigmoid()(logits)
        return prob


def train(model, train_tuple, loss_fn, opt, curr_epoch, ema):
    """ Train """
    # Shuffle train indices for current epoch, batching
    all_features, all_labels, all_ids, all_embed = train_tuple
    train_indices = list(range(len(all_labels)))

    shuffle(train_indices)
    train_features = all_features[train_indices]
    train_labels = all_labels[train_indices]
    train_embed = all_embed[train_indices]

    # switch to finetune
    if curr_epoch == 1:
        for g in opt.param_groups:
            g['lr'] = 1e-5

    model.train()
    with trange(0, len(train_indices), BATCH_SIZE,
                desc='Epoch {}'.format(curr_epoch)) as t:
        for batch_idx_start in t:
            opt.zero_grad()
            batch_idx_end = min(batch_idx_start + BATCH_SIZE, len(train_indices))

            batch_features = torch.tensor(train_features[batch_idx_start:batch_idx_end]).cuda()
            batch_labels = torch.tensor(train_labels[batch_idx_start:batch_idx_end]).float().cuda().unsqueeze(-1)
            batch_embed = torch.tensor(train_embed[batch_idx_start:batch_idx_end]).cuda()

            if curr_epoch < 1:
                preds = model(batch_features, batch_embed, freeze=True)
            else:
                preds = model(batch_features, batch_embed, freeze=False)
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
    val_features, val_labels, val_ids, val_embed = val_tuple

    model.eval()
    val_preds = []
    with torch.no_grad():
        for batch_idx_start in range(0, len(val_ids), BATCH_SIZE):
            batch_idx_end = min(batch_idx_start + BATCH_SIZE, len(val_ids))
            batch_features = torch.tensor(val_features[batch_idx_start:batch_idx_end]).cuda()
            batch_embed = torch.tensor(val_embed[batch_idx_start:batch_idx_end]).cuda()
            batch_preds = model(batch_features, batch_embed)
            val_preds.append(batch_preds.cpu())

        val_preds = np.concatenate(val_preds)
        val_roc_auc_score = roc_auc_score(val_labels, val_preds)
    return val_roc_auc_score


def main_driver(train_tuple, val_raw_tuple, val_translated_tuple):
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

    loss_fn = torch.nn.BCELoss()
    opt = torch.optim.Adam(classifier.parameters(), lr=1e-3)

    if USE_AMP:
        amp.register_float_function(torch, 'sigmoid')
        classifier, opt = amp.initialize(classifier, opt, opt_level='O1', verbosity=0)

    if USE_MULTI_GPU:
        classifier = torch.nn.DataParallel(classifier)

    best_raw_auc, best_translated_auc = -1, -1
    list_raw_auc, list_translated_auc = [], []
    for curr_epoch in range(NUM_EPOCHS):
        train(classifier, train_tuple, loss_fn, opt, curr_epoch, ema)

        epoch_raw_auc = evaluate(classifier, val_raw_tuple)
        epoch_translated_auc = evaluate(classifier, val_translated_tuple)
        print('Epoch {} - Raw: {:.4f}, Translated: {:.4f}'.format(curr_epoch, epoch_raw_auc, epoch_translated_auc))
        list_raw_auc.append(epoch_raw_auc)
        list_translated_auc.append(epoch_translated_auc)

        if epoch_translated_auc > best_translated_auc and SAVE_MODEL:
            print('Translated AUC increased; saving model')
            best_translated_auc = epoch_translated_auc
            save_model(os.path.join(OUTPUT_DIR, PRETRAINED_MODEL), classifier, pretrained_config, tokenizer)

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


def generate_use_embeddings(input_strings, batch_size=128):
    import tensorflow as tf
    import tensorflow_text
    embed = tf.saved_model.load(USE_MODEL_PATH)
    return np.concatenate([embed(train_strings[x:x + batch_size]).numpy()
                           for x in range(0, len(train_strings), batch_size)])


if __name__ == '__main__':
    start_time = time.time()

    train_ids, train_strings, train_labels = get_id_text_label_from_csv(TRAIN_CSV_PATH, text_col='comment_text')
    val_ids, val_raw_strings, val_labels = get_id_text_label_from_csv(VAL_CSV_PATH, text_col='comment_text')
    _, val_translated_strings, _ = get_id_text_label_from_csv(VAL_CSV_PATH, text_col='comment_text_en')

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

    print(train_features.shape, val_raw_features.shape, val_translated_features.shape)


    # Used to ensure TF graph is cleared
    with mp.Pool(1) as p:
        train_embed, val_raw_embed, val_translated_embed = \
            list(p.map(generate_use_embeddings, [train_strings, val_raw_strings, val_translated_strings]))

    print(train_embed.shape, val_raw_embed.shape, val_translated_embed.shape)

    main_driver([train_features, train_labels, train_ids, train_embed],
                [val_raw_features, val_labels, val_ids, val_raw_embed],
                [val_translated_features, val_labels, val_ids, val_translated_embed])

    print('ELapsed time: {}'.format(time.time() - start_time))
