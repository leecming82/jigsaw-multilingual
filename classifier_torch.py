"""
Baseline PyTorch classifier for Jigsaw Multilingual
"""
import time
import re
import pandas as pd
from itertools import starmap
from random import shuffle
from functools import partial
import multiprocessing as mp
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel, AutoConfig, WEIGHTS_NAME, CONFIG_NAME
from sklearn.metrics import roc_auc_score
from preprocessor import generate_train_kfolds_indices, get_id_text_label_from_csv
from postprocessor import score_roc_auc
from apex import amp

USE_AMP = True
PRETRAINED_MODEL = 'bert-base-multilingual-cased'
TRAIN_CSV_PATH = 'data/validation.csv'
OUTPUT_DIR = 'models/'
NUM_GPUS = 2  # Set to 1 if using AMP (doesn't seem to play nice with 1080 Ti)
MAX_CORES = 8  # limit MP calls to use this # cores at most
BASE_MODEL_OUTPUT_DIM = 768  # hidden layer dimensions
INTERMEDIATE_HIDDEN_UNITS = 1
MAX_SEQ_LEN = 200  # max sequence length for input strings: gets padded/truncated
NUM_EPOCHS = 4
BATCH_SIZE = 32


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


def train_driver(fold_id, fold_indices, all_features, all_labels, all_ids, gpu_id_queue):
    use_gpu_id = gpu_id_queue.get()
    fold_start_time = time.time()
    import os
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = str(use_gpu_id)
    print('Fold {} training: GPU_ID{}'.format(fold_id, use_gpu_id))
    pretrained_config = AutoConfig.from_pretrained(PRETRAINED_MODEL,
                                                   output_hidden_states=True)
    pretrained_base = AutoModel.from_pretrained(PRETRAINED_MODEL, config=pretrained_config).cuda()
    classifier = ClassifierHead(pretrained_base).cuda()

    loss_fn = torch.nn.BCELoss()
    opt = torch.optim.Adam(classifier.parameters(), lr=1e-3)

    if USE_AMP:
        amp.register_float_function(torch, 'sigmoid')
        classifier, opt = amp.initialize(classifier, opt, opt_level='O1', verbosity=0)

    train_indices, val_indices = fold_indices
    if fold_id is None:
        train_indices = np.concatenate([train_indices, val_indices])
    val_features, val_labels, val_ids = all_features[val_indices], all_labels[val_indices], all_ids[val_indices]

    epoch_eval_score = []
    epoch_val_id_to_pred = []
    for curr_epoch in range(NUM_EPOCHS):
        # Shuffle train indices for current epoch, batching
        shuffle(train_indices)
        train_features, train_labels = all_features[train_indices], all_labels[train_indices]

        # switch to finetune
        if curr_epoch == 1:
            for g in opt.param_groups:
                g['lr'] = 1e-5

        classifier.train()
        for batch_idx_start in range(0, len(train_indices), BATCH_SIZE):
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

        if fold_id is not None:
            # Evaluate validation fold
            classifier.eval()
            val_preds = []
            with torch.no_grad():
                for batch_idx_start in range(0, len(val_indices), BATCH_SIZE):
                    batch_idx_end = min(batch_idx_start + BATCH_SIZE, len(val_indices))
                    batch_features = torch.tensor(val_features[batch_idx_start:batch_idx_end]).cuda()
                    batch_preds = classifier(batch_features)
                    val_preds.append(batch_preds.cpu())

                val_preds = np.concatenate(val_preds)
                val_roc_auc_score = roc_auc_score(val_labels, val_preds)
                print('Fold {} eval score: {:.4f}'.format(fold_id, val_roc_auc_score))
                epoch_eval_score.append(val_roc_auc_score)
                epoch_val_id_to_pred.append({val_id: val_pred for val_id, val_pred in zip(val_ids, val_preds)})
        else:
            output_model_file = os.path.join(OUTPUT_DIR, WEIGHTS_NAME)
            output_config_file = os.path.join(OUTPUT_DIR, CONFIG_NAME)

            torch.save(classifier.state_dict(), output_model_file)
            pretrained_config.to_json_file(output_config_file)
            tokenizer.save_vocabulary(OUTPUT_DIR)
    gpu_id_queue.put(use_gpu_id)
    print('Fold {} run-time: {:.4f}'.format(fold_id, time.time() - fold_start_time))
    return epoch_eval_score, epoch_val_id_to_pred


if __name__ == '__main__':
    start_time = time.time()
    all_ids, all_strings, all_labels = get_id_text_label_from_csv(TRAIN_CSV_PATH)
    # all_strings = [x.lower() for x in all_strings]
    fold_indices = generate_train_kfolds_indices(all_strings)

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
        all_features = np.array(p.map(encode_partial, all_strings))

    print('Starting kfold training')
    with mp.Pool(NUM_GPUS, maxtasksperchild=1) as p:
        # prime GPU ID queue with IDs
        gpu_id_queue = mp.Manager().Queue()
        [gpu_id_queue.put(i) for i in range(NUM_GPUS)]

        results = p.starmap(train_driver,
                            ((fold_id,
                              curr_fold_indices,
                              all_features,
                              all_labels,
                              all_ids,
                              gpu_id_queue) for (fold_id, curr_fold_indices) in enumerate(fold_indices)))
    mean_score = np.mean(np.stack([x[0] for x in results]), axis=0)
    with np.printoptions(precision=4, suppress=True):
        print('Mean fold ROC_AUC_SCORE: {}'.format(mean_score))

    print('Generating/saving OOF predictions')
    oof_scores = []
    for curr_epoch in range(NUM_EPOCHS):
        oof_preds = {}
        [oof_preds.update(x[1][curr_epoch]) for x in results]
        oof_preds = pd.DataFrame.from_dict(oof_preds, orient='index').reset_index()
        oof_preds.columns = ['id', 'toxic']
        oof_out_path = 'data/oof/pt_{}_{}_{}.csv'.format(PRETRAINED_MODEL, curr_epoch + 1, MAX_SEQ_LEN)
        oof_preds.sort_values(by='id').to_csv(oof_out_path, index=False)
        oof_scores.append(score_roc_auc('data/validation.csv', oof_out_path))

    with np.printoptions(precision=4, suppress=True):
        print(np.array(oof_scores))

    train_driver(None, fold_indices[0], all_features, all_labels, all_ids, gpu_id_queue)
    print('Elapsed time: {}'.format(time.time() - start_time))
