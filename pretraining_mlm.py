"""
Base Torch classifier with MLM as aux loss
- BERT derivatives [CLS] [SEP]
- Camembert <s> </s>
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
from torch.utils.data import DataLoader, Dataset, RandomSampler
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer, AutoModelWithLMHead, AutoConfig
from tqdm import tqdm
from apex import amp
from torch_helpers import mask_tokens

PRETRAINED_MODEL = 'camembert-base'
TRAIN_FILE_PATH = 'data/fr_mlm.csv'
MODEL_PATH = 'models/fr_mlm'
NUM_EPOCHS = 30
BATCH_SIZE = 32
ACCUM_FOR = 1
LR = 1e-5


def cln(x):
    return ' '.join(x.split())


class TextDataset(Dataset):
    def __init__(self, tokenizer, train_file_path=TRAIN_FILE_PATH, block_size=512):
        """
        Read train CSV and either generate LAMBADA style features on the fly
        or if it exists, load from the pickle file
        """
        self.examples = []

        train_df = pd.read_csv(train_file_path)
        print('# training samples: {}'.format(train_df.shape[0]))
        raw_comments = [cln(x) for x in train_df['comment_text']]
        combined_comments = ''.join(['<s>{}</s>'.format(curr_comment) for curr_comment in raw_comments])
        combined_tokens = tokenizer.encode(combined_comments,
                                           add_special_tokens=False)
        # Truncate in block of block_size
        for i in range(0, len(combined_tokens) - block_size + 1, block_size):
            self.examples.append(combined_tokens[i: i + block_size])

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        return torch.tensor(self.examples[item])


def train_driver(train_dataset, lm_model, tokenizer):
    def collate(examples):
        """ collate fn passed to DataLoader"""
        if tokenizer._pad_token is None:
            return pad_sequence(examples, batch_first=True)
        return pad_sequence(examples, batch_first=True, padding_value=tokenizer.pad_token_id)

    lm_model = lm_model.cuda()
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset,
                                  sampler=train_sampler,
                                  batch_size=BATCH_SIZE,
                                  collate_fn=collate)

    opt = torch.optim.Adam(lm_model.parameters(), lr=LR)
    lm_model, opt = amp.initialize(lm_model, opt, opt_level='O1', verbosity=0)
    lm_model = torch.nn.DataParallel(lm_model)

    lm_model.train()
    for curr_epoch in range(NUM_EPOCHS):
        iter = 0
        running_total_loss = 0
        with tqdm(train_dataloader) as t:
            for step, batch in enumerate(t):
                iter += 1
                t.set_description('Epoch {}/{}'.format(curr_epoch + 1, NUM_EPOCHS))
                batch_features = batch.cuda()
                batch_features, masked_lm_labels = mask_tokens(batch_features, tokenizer)

                outputs = lm_model(input_ids=batch_features,
                                   masked_lm_labels=masked_lm_labels)
                loss = outputs[0]
                loss = loss.mean()
                loss = loss / ACCUM_FOR

                with amp.scale_loss(loss, opt) as scaled_loss:
                    scaled_loss.backward()

                running_total_loss += loss.detach().cpu().numpy()
                t.set_postfix(loss=running_total_loss / iter)

                if iter % ACCUM_FOR == 0:
                    opt.step()
                    opt.zero_grad()

        model_output_path = os.path.join(MODEL_PATH, PRETRAINED_MODEL, str(curr_epoch))
        if not os.path.exists(model_output_path):
            os.makedirs(model_output_path)

        lm_model.module.save_pretrained(model_output_path)
        tokenizer.save_pretrained(model_output_path)


if __name__ == '__main__':
    tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_MODEL)
    print('Mask ID: {}, Pad ID: {}'.format(tokenizer.mask_token_id, tokenizer.pad_token_id))
    lm_model = AutoModelWithLMHead.from_pretrained(PRETRAINED_MODEL)
    data = TextDataset(tokenizer)
    train_driver(data, lm_model, tokenizer)
