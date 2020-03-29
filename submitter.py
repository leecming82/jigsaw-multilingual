import os
import time
import multiprocessing as mp
from functools import partial
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel, AutoConfig, WEIGHTS_NAME

TRAINED_MODEL_PATH = 'models/bert-base-multilingual-cased'
TEST_CSV_PATH_1 = 'data/test_en.csv'
TEST_CSV_PATH_2 = 'data/jigsaw_miltilingual_test_translated.csv'
SUBMIT_CSV_PATH = 'data/test/bert-base-multilingual-cased.csv'
MAX_CORES = 24
MAX_SEQ_LEN = 200
BATCH_SIZE = 64
BASE_MODEL_OUTPUT_DIM = 768  # hidden layer dimensions
INTERMEDIATE_HIDDEN_UNITS = 1

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def get_id_text_from_test_csv(csv_path, text_col):
    """
    Load training data
    :param csv_path: path of csv with 'id' 'comment_text' columns present
    :return:
    """
    raw_pdf = pd.read_csv(csv_path)
    return raw_pdf['id'].values, list(raw_pdf[text_col].values)


class ClassifierHead(torch.nn.Module):
    def __init__(self, base_model):
        super(ClassifierHead, self).__init__()
        self.base_model = base_model
        self.cnn = torch.nn.Conv1d(BASE_MODEL_OUTPUT_DIM, INTERMEDIATE_HIDDEN_UNITS, kernel_size=1)
        self.fc = torch.nn.Linear(BASE_MODEL_OUTPUT_DIM, INTERMEDIATE_HIDDEN_UNITS)

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


def predict(model, input_features):
    print('Predicting test comments')
    preds = []
    model.eval()
    with torch.no_grad():
        for batch_idx_start in range(0, len(input_features), BATCH_SIZE):
            batch_idx_end = min(batch_idx_start + BATCH_SIZE, len(input_features))
            batch_features = torch.tensor(input_features[batch_idx_start:batch_idx_end]).cuda()
            batch_preds = model(batch_features)
            preds.append(batch_preds.cpu())

    return np.concatenate(preds).squeeze()


start_time = time.time()
all_ids, all_strings_1 = get_id_text_from_test_csv(TEST_CSV_PATH_1, text_col='content_en')
_, all_strings_2 = get_id_text_from_test_csv(TEST_CSV_PATH_2, text_col='translated')

assert len(all_strings_1) == len(all_strings_2)

tokenizer = AutoTokenizer.from_pretrained(TRAINED_MODEL_PATH)
encode_partial = partial(tokenizer.encode,
                         max_length=MAX_SEQ_LEN,
                         pad_to_max_length=True,
                         add_special_tokens=True)
print('Encoding raw strings into model-specific tokens')
with mp.Pool(MAX_CORES) as p:
    features = [np.array(p.map(encode_partial, all_strings_1)),
                np.array(p.map(encode_partial, all_strings_2))]

pretrained_config = AutoConfig.from_pretrained(TRAINED_MODEL_PATH,
                                               output_hidden_states=True)
pretrained_base = AutoModel.from_config(pretrained_config)
classifier = ClassifierHead(pretrained_base).cuda()
classifier.load_state_dict(torch.load(os.path.join(TRAINED_MODEL_PATH, WEIGHTS_NAME)))

test_preds = np.mean(np.stack([predict(classifier, x) for x in features]), 0)

pd.DataFrame({'id': all_ids, 'toxic': test_preds}).to_csv(SUBMIT_CSV_PATH, index=False)
print('Elapsed time: {}'.format(time.time() - start_time))
