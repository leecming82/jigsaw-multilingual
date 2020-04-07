"""
Mod of baseline to use pairs to classify (raw + en translation)
"""
import os
import time
from itertools import starmap
from random import shuffle
from functools import partial
import multiprocessing as mp
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel, AutoConfig, WEIGHTS_NAME
from sklearn.metrics import roc_auc_score
from apex import amp
from tqdm import trange
from preprocessor import get_id_text_label_from_csv
from torch_helpers import save_model

SAVE_MODEL = True
USE_AMP = True
USE_MULTI_GPU = False
PRETRAINED_MODEL = 'models/train3_base_en'
VAL_CSV_PATH = 'data/validation_en.csv'
OUTPUT_DIR = 'models/pairs'
NUM_GPUS = 2  # Set to 1 if using AMP (doesn't seem to play nice with 1080 Ti)
MAX_CORES = 24  # limit MP calls to use this # cores at most
BASE_MODEL_OUTPUT_DIM = 1024  # hidden layer dimensions
INTERMEDIATE_HIDDEN_UNITS = 1
MAX_SEQ_LEN = 200  # max sequence length for input strings: gets padded/truncated
NUM_EPOCHS = 3  # Half trained using train, half on val (+ PL)
BATCH_SIZE = 24
ACCUM_FOR = 2
SAVE_ITER = 100  # save every X iterations
LR = 1e-5

if not USE_MULTI_GPU:
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
        self.cnn = torch.nn.Conv1d(BASE_MODEL_OUTPUT_DIM, INTERMEDIATE_HIDDEN_UNITS, kernel_size=1)
        self.fc = torch.nn.Linear(BASE_MODEL_OUTPUT_DIM, INTERMEDIATE_HIDDEN_UNITS)

    def forward(self, raw_features, translated_features):
        hidden_states_raw = self.base_model(raw_features)[0]
        with torch.no_grad():
            hidden_states_translated = self.base_model(translated_features)[0]

        hidden_states = torch.cat([hidden_states_raw, hidden_states_translated], dim=1)

        hidden_states = hidden_states.permute(0, 2, 1)
        cnn_states = self.cnn(hidden_states)
        cnn_states = cnn_states.permute(0, 2, 1)
        logits, _ = torch.max(cnn_states, 1)

        prob = torch.nn.Sigmoid()(logits)
        return prob


def train(model, config, train_tuple, loss_fn, opt, curr_epoch):
    """ Train """
    # Shuffle train indices for current epoch, batching
    all_raw_features, all_translated_features, all_labels, all_ids = train_tuple
    train_indices = list(range(len(all_labels)))

    shuffle(train_indices)
    train_raw_features = all_raw_features[train_indices]
    train_translated_features = all_translated_features[train_indices]
    train_labels = all_labels[train_indices]

    model.train()
    iter = 0
    running_total_loss = 0
    with trange(0, len(train_indices), BATCH_SIZE,
                desc='Epoch {}'.format(curr_epoch)) as t:
        for batch_idx_start in t:
            iter += 1
            batch_idx_end = min(batch_idx_start + BATCH_SIZE, len(train_indices))

            batch_raw_features = torch.tensor(train_raw_features[batch_idx_start:batch_idx_end]).cuda()
            batch_translated_features = torch.tensor(train_translated_features[batch_idx_start:batch_idx_end]).cuda()
            batch_labels = torch.tensor(train_labels[batch_idx_start:batch_idx_end]).float().cuda().unsqueeze(-1)

            preds = model(batch_raw_features, batch_translated_features)
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

            # Save every specified step on last epoch
            if curr_epoch == NUM_EPOCHS - 1 and iter % SAVE_ITER == 0 and SAVE_MODEL:
                print('Saving epoch {} model'.format(curr_epoch))
                save_model(os.path.join(OUTPUT_DIR, PRETRAINED_MODEL, '{}_{}'.format(curr_epoch, iter)),
                           model,
                           config,
                           tokenizer)


def evaluate(model, val_tuple):
    # Evaluate validation AUC
    val_raw_features, val_translated_features, val_labels, val_ids = val_tuple

    model.eval()
    val_preds = []
    with torch.no_grad():
        for batch_idx_start in range(0, len(val_ids), BATCH_SIZE):
            batch_idx_end = min(batch_idx_start + BATCH_SIZE, len(val_ids))
            batch_raw_features = torch.tensor(val_raw_features[batch_idx_start:batch_idx_end]).cuda()
            batch_translated_features = torch.tensor(val_translated_features[batch_idx_start:batch_idx_end]).cuda()
            batch_preds = model(batch_raw_features, batch_translated_features)
            val_preds.append(batch_preds.cpu())

        val_preds = np.concatenate(val_preds)
        val_roc_auc_score = roc_auc_score(val_labels, val_preds)
    return val_roc_auc_score


def main_driver(train_tuple):
    pretrained_config = AutoConfig.from_pretrained(PRETRAINED_MODEL,
                                                   output_hidden_states=True)
    pretrained_base = AutoModel.from_pretrained(PRETRAINED_MODEL, config=pretrained_config).cuda()
    classifier = ClassifierHead(pretrained_base).cuda()
    classifier.load_state_dict(torch.load(os.path.join(PRETRAINED_MODEL, WEIGHTS_NAME)))

    loss_fn = torch.nn.BCELoss()
    opt = torch.optim.Adam(classifier.parameters(), lr=LR)

    if USE_AMP:
        amp.register_float_function(torch, 'sigmoid')
        classifier, opt = amp.initialize(classifier, opt, opt_level='O1', verbosity=0)

    if USE_MULTI_GPU:
        classifier = torch.nn.DataParallel(classifier)

    list_auc = []
    for curr_epoch in range(NUM_EPOCHS):
        # switch to finetune - only lower for those > finetune LR
        # i.e., lower layers might have even smaller LR
        train(classifier, pretrained_config, train_tuple, loss_fn, opt, curr_epoch)

        epoch_auc = evaluate(classifier, train_tuple)
        print('Epoch {} - Raw: {:.4f}'.format(curr_epoch, epoch_auc))
        list_auc.append(epoch_auc)

    with np.printoptions(precision=4, suppress=True):
        print(np.array(list_auc))


if __name__ == '__main__':
    start_time = time.time()
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
        val_raw_features = np.array(p.map(encode_partial, val_raw_strings))
        val_translated_features = np.array(p.map(encode_partial, val_translated_strings))

    main_driver([val_raw_features, val_translated_features, val_labels, val_ids])

    print('Elapsed time: {}'.format(time.time() - start_time))
