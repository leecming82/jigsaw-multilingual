"""
Baseline PyTorch classifier using a pretrained HuggingFace Transformer model
- Run prepare_data.py prior to generate the prerequisite training files
- Classifier head on-top of the 1st token of the last hidden layer from the base pretrained model
- Uses APEX mixed precision (FP16) training
- Allows for gradient accumulation with the ACCUM_FOR flag
- checkpoint ensembling by predicting the test-set every epoch (saves to $PREDICTION_DIR/{EPOCH_NUM}.csv)
- Given NUM_EPOCHS, trains against the train dataset for half the epochs, and the val dataset for remaining half
- for preprocessing comments, truncates adjacent whitespaces to a single whitespace
"""
import json
import os
import time
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

with open('SETTINGS.json') as f:
    SETTINGS_DICT = json.load(f)

PRETRAINED_MODEL = 'mrm8488/distill-bert-base-spanish-wwm-cased-finetuned-spa-squad2-es'
TRAIN_CSV_PATH = os.path.join(SETTINGS_DICT['TRAIN_DATA_DIR'], 'curr_run_train.csv')
TEST_CSV_PATH = os.path.join(SETTINGS_DICT['TRAIN_DATA_DIR'], 'curr_run_test.csv')
VAL_CSV_PATH = os.path.join(SETTINGS_DICT['TRAIN_DATA_DIR'], 'curr_run_val.csv')
MAX_CORES = 24  # limit MP calls to use this # cores at most; for tokenizing
BASE_MODEL_OUTPUT_DIM = 768  # hidden layer dimensions
NUM_OUTPUTS = 1  # Num of output units (should be 1 for Toxicity)
MAX_SEQ_LEN = 200  # max sequence length for input strings: gets padded/truncated
# Num. epochs to train against (if validation data exists, the model will switch to training against the validation
# data in the 2nd half of epochs
NUM_EPOCHS = 6
# Gradient Accumulation: updates every ACCUM_FOR steps so that effective BS = BATCH_SIZE * ACCUM_FOR
BATCH_SIZE = 64
ACCUM_FOR = 1
LR = 1e-5  # Learning rate - constant value

# For multi-gpu environments - make only 1 GPU visible to process
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


class ClassifierHead(torch.nn.Module):
    """
    Bert base with a Linear layer plopped on top of it
    - connects the CLS token of the last hidden layer with the FC
    """

    def __init__(self, base_model):
        super(ClassifierHead, self).__init__()
        self.base_model = base_model
        self.cnn = torch.nn.Conv1d(BASE_MODEL_OUTPUT_DIM, NUM_OUTPUTS, kernel_size=1)
        self.fc = torch.nn.Linear(BASE_MODEL_OUTPUT_DIM, NUM_OUTPUTS)

    def forward(self, x):
        hidden_states = self.base_model(x)[0]

        # If you want to max-pool on a CNN of all tokens of the last hidden layer
        # hidden_states = hidden_states.permute(0, 2, 1)
        # cnn_states = self.cnn(hidden_states)
        # cnn_states = cnn_states.permute(0, 2, 1)
        # logits, _ = torch.max(cnn_states, 1)

        # FC on 1st token (typically CLS special token)
        logits = self.fc(hidden_states[:, 0, :])
        prob = torch.nn.Sigmoid()(logits)
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

            batch_features = torch.tensor(train_features[batch_idx_start:batch_idx_end]).cuda()
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
            batch_features = torch.tensor(data_tuple[0][batch_idx_start:batch_idx_end]).cuda()
            batch_preds = model(batch_features)
            val_preds.append(batch_preds.cpu().numpy().reshape(-1))

    val_preds = np.concatenate(val_preds)
    if score:
        # predict validation samples
        val_score = roc_auc_score(np.round(data_tuple[1]), val_preds)
    else:
        # predicting test samples
        curr_test_path = os.path.join(SETTINGS_DICT['PREDICTION_DIR'],
                                      '{}.csv'.format(epoch))
        pd.DataFrame({'id': data_tuple[-1], 'toxic': val_preds}) \
            .to_csv(curr_test_path, index=False)

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
    list_auc = []

    current_tuple = train_tuple
    for curr_epoch in range(NUM_EPOCHS):
        # After half epochs, switch to training against validation set
        if curr_epoch == NUM_EPOCHS // 2 and len(val_tuple[-1]) > 0:
            current_tuple = val_tuple
        train(classifier, current_tuple, loss_fn, opt, curr_epoch)

        # Score against the validation set
        if len(val_tuple[-1]) > 0:
            epoch_raw_auc = predict_evaluate(classifier, val_tuple, curr_epoch, score=True)
            print('Epoch {} - Val AUC: {:.4f}'.format(curr_epoch, epoch_raw_auc))
            list_auc.append(epoch_raw_auc)

        predict_evaluate(classifier, test_tuple, curr_epoch)

    with np.printoptions(precision=4, suppress=True):
        print(np.array(list_auc))


if __name__ == '__main__':
    def cln(x):  # Truncates adjacent whitespaces to single whitespace
        return ' '.join(x.split())


    start_time = time.time()

    # Load train, validation, and pseudo-label data
    train_ids, train_strings, train_labels = get_id_text_label_from_csv(TRAIN_CSV_PATH,
                                                                        text_col='comment_text')
    train_strings = [cln(x) for x in train_strings]

    val_ids, val_strings, val_labels = get_id_text_label_from_csv(VAL_CSV_PATH,
                                                                  text_col='comment_text')
    val_strings = [cln(x) for x in val_strings]

    test_ids, test_strings = get_id_text_from_test_csv(TEST_CSV_PATH, text_col='comment_text')
    test_strings = [cln(x) for x in test_strings]

    # use MP to batch encode the raw feature strings into Bert token IDs
    tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_MODEL)
    encode_partial = partial(tokenizer.encode,
                             truncation=True,
                             max_length=MAX_SEQ_LEN,
                             pad_to_max_length=True,
                             add_special_tokens=True)
    print('Encoding raw strings into model-specific tokens')
    with mp.Pool(MAX_CORES) as p:
        train_features = np.array(p.map(encode_partial, train_strings))
        val_features = np.array(p.map(encode_partial, val_strings))
        test_features = np.array(p.map(encode_partial, test_strings))

    print('Train size: {}, val size: {}'.format(len(train_ids), len(val_ids)))
    print('Train positives: {}, train negatives: {}'.format(train_labels[train_labels == 1].shape,
                                                            train_labels[train_labels == 0].shape))

    main_driver([train_features, train_labels, train_ids],
                [val_features, val_labels, val_ids],
                [test_features, test_ids],
                tokenizer)

    print('Elapsed time: {}'.format(time.time() - start_time))
