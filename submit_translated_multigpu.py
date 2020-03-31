# Uses multi-gpu to predict list of test data
import time
import multiprocessing as mp
from functools import partial
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel, AutoConfig, WEIGHTS_NAME

TRAINED_MODEL_PATH = 'models/train_val_pl/xlm-roberta-large/2'
TEST_SETS = [['data/test_en.csv', 'content'],
             ['data/test_en.csv', 'content_en'],
             ['data/jigsaw_miltilingual_test_translated.csv', 'translated']]
SUBMIT_CSV_PATH = 'data/submission_xlm2_train_val_pl9317_{}.csv'
NUM_GPUS = 2
MAX_CORES = 24
MAX_SEQ_LEN = 200
BATCH_SIZE = 24
BASE_MODEL_OUTPUT_DIM = 1024  # hidden layer dimensions
INTERMEDIATE_HIDDEN_UNITS = 1


def get_id_text_from_test_csv(csv_path, text_col):
    """
    Load training data
    :param csv_path: path of csv with 'id' 'comment_text' columns present
    :return:
    """
    raw_pdf = pd.read_csv(csv_path)
    return raw_pdf['id'].values, list(raw_pdf[text_col].values)


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

        # logits = self.fc(hidden_states[:, -1, :])
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


def main_driver(input_features, gpu_id_queue):
    use_gpu_id = gpu_id_queue.get()
    import os
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = str(use_gpu_id)
    print('Predicting on GPU_ID{}'.format(use_gpu_id))

    pretrained_config = AutoConfig.from_pretrained(TRAINED_MODEL_PATH,
                                                   output_hidden_states=True)
    pretrained_base = AutoModel.from_config(pretrained_config)
    classifier = ClassifierHead(pretrained_base).cuda()
    classifier.load_state_dict(torch.load(os.path.join(TRAINED_MODEL_PATH, WEIGHTS_NAME)))

    gpu_id_queue.put(use_gpu_id)
    return predict(classifier, input_features)


start_time = time.time()
test_set_tuples = [get_id_text_from_test_csv(curr_test_set[0], curr_test_set[1]) for curr_test_set in TEST_SETS]

tokenizer = AutoTokenizer.from_pretrained(TRAINED_MODEL_PATH)
encode_partial = partial(tokenizer.encode,
                         max_length=MAX_SEQ_LEN,
                         pad_to_max_length=True,
                         add_special_tokens=True)
print('Encoding raw strings into model-specific tokens')
with mp.Pool(MAX_CORES) as p:
    features = [np.array(p.map(encode_partial, curr_test_set_tuple[1])) for curr_test_set_tuple in test_set_tuples]

with mp.Pool(NUM_GPUS, maxtasksperchild=1) as p:
    # prime GPU ID queue with IDs
    gpu_id_queue = mp.Manager().Queue()
    [gpu_id_queue.put(i) for i in range(NUM_GPUS)]

    results = p.starmap(main_driver,
                        ((curr_feature,
                          gpu_id_queue) for curr_feature in features))

print('Generating {} sets of predictions'.format(len(results)))
for i, curr_results in enumerate(results):
    pd.DataFrame({'id': test_set_tuples[0][0], 'toxic': curr_results}). \
        to_csv(SUBMIT_CSV_PATH.format(i), index=False)

avg_preds = np.mean(np.stack(results), 0)
pd.DataFrame({'id': test_set_tuples[0][0], 'toxic': avg_preds}) \
    .to_csv(SUBMIT_CSV_PATH.format('avg'), index=False)

print('Elapsed time: {}'.format(time.time() - start_time))
