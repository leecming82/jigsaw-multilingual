"""
Baseline PyTorch classifier for Jigsaw Multilingual
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
from preprocessor import (get_id_text_label_from_csv,
                          get_id_text_label_from_csvs)
from apex import amp
from tqdm import trange

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

USE_AMP = True
PRETRAINED_MODEL = 'xlm-roberta-base'
OUTPUT_DIR = 'models/'
MAX_CORES = 24  # limit MP calls to use this # cores at most
BASE_MODEL_OUTPUT_DIM = 768  # hidden layer dimensions
INTERMEDIATE_HIDDEN_UNITS = 1
MAX_SEQ_LEN = 200  # max sequence length for input strings: gets padded/truncated
NUM_EPOCHS = 3
BATCH_SIZE = 64


class ClassifierHead(torch.nn.Module):
    """
    Bert base with a Linear layer plopped on top of it
    - connects the max pool of the last hidden layer with the FC
    """

    def __init__(self, base_model):
        super(ClassifierHead, self).__init__()
        self.base_model = base_model
        self.cnn = torch.nn.Conv1d(BASE_MODEL_OUTPUT_DIM, INTERMEDIATE_HIDDEN_UNITS, kernel_size=1)

    def forward(self, x, freeze=True):
        if freeze:
            with torch.no_grad():
                hidden_states = self.base_model(x)[0]
        else:
            hidden_states = self.base_model(x)[0]

        hidden_states = hidden_states.permute(0, 2, 1)
        cnn_states = self.cnn(hidden_states)
        cnn_states = cnn_states.permute(0, 2, 1)
        logits, _ = torch.max(cnn_states, 1)

        prob = torch.nn.Sigmoid()(logits)
        return prob


def train_driver(train_tuple, val_tuple):
    pretrained_config = AutoConfig.from_pretrained(PRETRAINED_MODEL,
                                                   output_hidden_states=True)
    pretrained_base = AutoModel.from_pretrained(PRETRAINED_MODEL, config=pretrained_config).cuda()
    classifier = ClassifierHead(pretrained_base).cuda()

    loss_fn = torch.nn.BCELoss()
    opt = torch.optim.Adam(classifier.parameters(), lr=1e-3)

    if USE_AMP:
        amp.register_float_function(torch, 'sigmoid')
        classifier, opt = amp.initialize(classifier, opt, opt_level='O1', verbosity=0)

    # classifier = torch.nn.DataParallel(classifier)

    train_ids, train_features, train_labels = train_tuple
    val_ids, val_features, val_labels = val_tuple
    train_indices = list(range(len(train_ids)))

    epoch_eval_score = []
    best_eval_score = -1
    for curr_epoch in range(NUM_EPOCHS):
        # Shuffle train indices for current epoch, batching
        shuffle(train_indices)
        train_features, train_labels = train_features[train_indices], train_labels[train_indices]

        # switch to finetune
        if curr_epoch == 1:
            for g in opt.param_groups:
                g['lr'] = 1e-5

        classifier.train()
        with trange(0, len(train_features), BATCH_SIZE, desc=str(curr_epoch)) as t:
            for batch_idx_start in t:
                opt.zero_grad()
                batch_idx_end = min(batch_idx_start + BATCH_SIZE, len(train_indices))

                batch_features = torch.tensor(train_features[batch_idx_start:batch_idx_end]).cuda()
                batch_labels = torch.tensor(train_labels[batch_idx_start:batch_idx_end]).float().cuda().unsqueeze(-1)

                if curr_epoch < 1:
                    preds = classifier(batch_features, freeze=True)
                else:
                    preds = classifier(batch_features, freeze=False)
                loss = loss_fn(preds, batch_labels)

                if USE_AMP:
                    with amp.scale_loss(loss, opt) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()

                opt.step()

        # Evaluate validation fold
        classifier.eval()
        val_preds = []
        with torch.no_grad():
            for batch_idx_start in range(0, len(val_ids), BATCH_SIZE):
                batch_idx_end = min(batch_idx_start + BATCH_SIZE, len(val_ids))
                batch_features = torch.tensor(val_features[batch_idx_start:batch_idx_end]).cuda()
                batch_preds = classifier(batch_features)
                val_preds.append(batch_preds.cpu())

            val_preds = np.concatenate(val_preds)
            val_roc_auc_score = roc_auc_score(val_labels, val_preds)
            epoch_eval_score.append(val_roc_auc_score)
            print('Epoch {} eval score: {:.4f}'.format(curr_epoch, val_roc_auc_score))

        if val_roc_auc_score > best_eval_score:
            best_eval_score = val_roc_auc_score
            print('Saving on epoch {}'.format(curr_epoch))
            save_path = os.path.join(OUTPUT_DIR, PRETRAINED_MODEL)
            if not os.path.exists(save_path):
                os.makedirs(save_path)

            output_model_file = os.path.join(save_path, WEIGHTS_NAME)
            output_config_file = os.path.join(save_path, CONFIG_NAME)

            torch.save(classifier.state_dict(), output_model_file)
            pretrained_config.to_json_file(output_config_file)
            tokenizer.save_vocabulary(save_path)

    return epoch_eval_score


if __name__ == '__main__':
    start_time = time.time()
    train_ids, train_strings, train_labels = get_id_text_label_from_csv('data/jigsaw-toxic-comment-train.csv')
    val_ids, val_strings, val_labels = get_id_text_label_from_csv('data/validation.csv')

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
        val_features = np.array(p.map(encode_partial, val_strings))
    print('Training data shape: {}'.format(train_features.shape))
    print('Validation data shape: {}'.format(val_features.shape))

    print('Starting training')
    epoch_eval_score = train_driver([train_ids, train_features, train_labels],
                                    [val_ids, val_features, val_labels])
    print(epoch_eval_score)
    print('Elapsed time: {}'.format(time.time() - start_time))
