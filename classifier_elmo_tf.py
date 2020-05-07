"""
Use pretrained multilingual ELMo from https://github.com/HIT-SCIR/ELMoForManyLangs
to train a BiGRU model
- very ugly code w/ the ELMo library using torch and the BiGRU model using tf.keras
- too large to pre-generate features so generate on the fly using a keras generator
- ELMo torch code on GPU while force BiGRU model on CPU
"""
import math
from sklearn.utils import shuffle
import time
import multiprocessing as mp
import pandas as pd
import numpy as np
import torch
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.layers as layers
from tensorflow.keras import Model
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from tensorflow.python.keras.preprocessing.text import text_to_word_sequence
from sklearn.metrics import roc_auc_score
from preprocessor import get_id_text_label_from_csv, get_id_text_from_test_csv
from elmoformanylangs import Embedder

RUN_NAME = '9544pt_elmo'
USE_LANG = 'pt'
ELMO_MODEL_PATH = 'models/elmo-pt'
TRAIN_CSV_PATH = 'data/pt_all.csv'
TRAIN_SAMPLE_FRAC = 1.
TEST_CSV_PATH = 'data/pt_test.csv'
VAL_CSV_PATH = 'data/validation_en.csv'
NUM_OUTPUTS = 1  # Number of targets
MAX_SEQ_LEN = 200  # max sequence length for input strings: gets padded/truncated
NUM_EPOCHS = 4
BATCH_SIZE = 64
NUM_CORES = 12
EMBEDDING_DIMS = 1024
HIDDEN_UNITS = 128


def tokenize(text):
    """ modify text_to_word_sequence to return MAX_SEQ_LEN max words """
    return text_to_word_sequence(text, lower=False)[:MAX_SEQ_LEN]


class ELMoGenerator:
    """ generate ELMo features on the fly"""
    def __init__(self, data_x, data_y, shuffle=False):
        self.data_x = data_x
        self.data_y = data_y
        self.embedder = Embedder(ELMO_MODEL_PATH, batch_size=BATCH_SIZE)
        self.shuffle = shuffle

    def batch_generator(self):
        while True:
            idx_list = list(range(len(self.data_x)))
            if self.shuffle:
                self.data_x, self.data_y = shuffle(self.data_x, self.data_y)

            for batch_idx_start in range(0, len(idx_list), BATCH_SIZE):
                batch_idx_end = min(len(idx_list), batch_idx_start + BATCH_SIZE)
                batch_features = self.embedder.sents2elmo(self.data_x[batch_idx_start:batch_idx_end])
                batch_features = pad_sequences(batch_features,
                                               maxlen=MAX_SEQ_LEN,
                                               padding='post',
                                               truncating='post',
                                               dtype='float32')

                if self.data_y is not None:
                    batch_labels = self.data_y[batch_idx_start:batch_idx_end]
                    yield batch_features, batch_labels
                else:
                    yield batch_features


def build_classifier_model():
    input = layers.Input(shape=(MAX_SEQ_LEN, EMBEDDING_DIMS), dtype=np.float32)
    gru_output = layers.Bidirectional(layers.GRU(HIDDEN_UNITS,
                                                 return_sequences=True))(input)
    gru_output = layers.Bidirectional(layers.GRU(HIDDEN_UNITS))(gru_output)
    prob = layers.Dense(NUM_OUTPUTS, activation='sigmoid')(gru_output)

    bigru_model = Model(input, prob)
    print('generated bigru model...')
    return bigru_model


def train_driver(train_tuple,
                 val_tuple,
                 test_tuple):
    train_strings, train_labels = train_tuple
    val_strings, val_labels = val_tuple
    test_strings, test_ids = test_tuple

    with tf.device('/cpu:0'):
        classifier = build_classifier_model()
        opt = Adam()
        opt = tf.keras.mixed_precision.experimental.LossScaleOptimizer(opt, 'dynamic')
        classifier.compile(optimizer=opt, loss='binary_crossentropy')

        list_val_auc = []
        for curr_epoch in range(NUM_EPOCHS):
            # Need to recreate generators each epoch
            torch.cuda.empty_cache()
            train_generator = ELMoGenerator(train_strings, train_labels, shuffle=True).batch_generator()
            val_generator = ELMoGenerator(val_strings, val_labels, shuffle=False).batch_generator()
            test_generator = ELMoGenerator(test_strings, None, shuffle=False).batch_generator()

            classifier.fit(train_generator,
                           steps_per_epoch=math.ceil(len(train_labels) / BATCH_SIZE),
                           epochs=1,
                           verbose=1)

            if len(val_labels):
                val_preds = classifier.predict(val_generator,
                                               steps=math.ceil(len(val_labels) / BATCH_SIZE))
                val_roc_auc_score = roc_auc_score(val_labels, val_preds)
                print(val_roc_auc_score)
                list_val_auc.append(val_roc_auc_score)

            test_preds = classifier.predict(test_generator,
                                            steps=math.ceil(len(test_strings) / BATCH_SIZE)).squeeze()
            pd.DataFrame({'id': test_ids, 'toxic': test_preds}) \
                .to_csv('data/outputs/test/{}_{}.csv'.format(RUN_NAME,
                                                             curr_epoch),
                        index=False)

        print(list_val_auc)


if __name__ == '__main__':
    start_time = time.time()

    # # Load train, validation, and test data
    train_ids, train_strings, train_labels = get_id_text_label_from_csv(TRAIN_CSV_PATH,
                                                                        text_col='comment_text',
                                                                        sample_frac=TRAIN_SAMPLE_FRAC)
    val_ids, val_strings, val_labels = get_id_text_label_from_csv(VAL_CSV_PATH,
                                                                  text_col='comment_text',
                                                                  lang=USE_LANG)
    test_ids, test_strings = get_id_text_from_test_csv(TEST_CSV_PATH, text_col='comment_text')

    with mp.Pool(NUM_CORES) as p:
        list_train_words = list(p.map(tokenize, train_strings))
        list_val_words = list(p.map(tokenize, val_strings))
        list_test_words = list(p.map(tokenize, test_strings))

    print(len(list_train_words), len(list_val_words), len(list_test_words))

    train_driver([list_train_words, train_labels],
                 [list_val_words, val_labels],
                 [list_test_words, test_ids])

    print('Elapsed time: {}'.format(time.time() - start_time))
