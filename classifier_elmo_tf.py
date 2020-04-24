"""
Draft code for training using ELMO multilingual features
"""
import time
from itertools import starmap
from functools import partial
import multiprocessing as mp
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.layers as layers
from tensorflow.keras import Model
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from tensorflow.python.keras.preprocessing.text import text_to_word_sequence
from sklearn.metrics import roc_auc_score
from preprocessor import get_id_text_label_from_csv, get_id_text_from_test_csv
from elmoformanylangs import Embedder

TRAIN_CSV_PATH = 'data/it_all.csv'
TRAIN_SAMPLE_FRAC = 1.
TEST_CSV_PATH = 'data/it_test.csv'
VAL_CSV_PATH = 'data/validation_en.csv'
NUM_OUTPUTS = 1  # Number of targets
MAX_SEQ_LEN = 200  # max sequence length for input strings: gets padded/truncated
NUM_EPOCHS = 4
BATCH_SIZE = 32
NUM_CORES = 12
EMBEDDING_DIMS = 1024
HIDDEN_UNITS = 128

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
                 embedding_matrix):
    train_features, train_labels = train_tuple
    val_features, val_labels = val_tuple

    classifier = build_classifier_model(embedding_matrix)
    opt = Adam()
    opt = tf.keras.mixed_precision.experimental.LossScaleOptimizer(opt, 'dynamic')
    classifier.compile(optimizer=opt, loss='binary_crossentropy')

    for _ in range(NUM_EPOCHS):
        classifier.fit(train_features, train_labels,
                       batch_size=BATCH_SIZE,
                       epochs=1,
                       verbose=1)
        val_preds = classifier.predict(val_features)
        val_roc_auc_score = roc_auc_score(val_labels, val_preds)
        print(val_roc_auc_score)


if __name__ == '__main__':
    start_time = time.time()
    e = Embedder('models/elmo-it')
    # list_embedded = e.sents2elmo([['hello', 'how', 'are', 'you']])

    # # Load train, validation, and pseudo-label data
    train_ids, train_strings, train_labels = get_id_text_label_from_csv(TRAIN_CSV_PATH,
                                                                        text_col='comment_text',
                                                                        sample_frac=TRAIN_SAMPLE_FRAC)
    val_ids, val_strings, val_labels = get_id_text_label_from_csv(VAL_CSV_PATH,
                                                                  text_col='comment_text',
                                                                  lang='it')
    test_ids, test_strings = get_id_text_from_test_csv(TEST_CSV_PATH, text_col='comment_text')

    tokenize = partial(text_to_word_sequence, lower=False)
    with mp.Pool(NUM_CORES) as p:
        list_train_words = list(p.map(tokenize, train_strings))

    list_train_tokens = e.sents2elmo(list_train_words)
    train_features = pad_sequences(list_train_tokens, maxlen=MAX_SEQ_LEN, padding='post', truncating='post')
    print(train_features.shape)
    print(train_features[0])

    # print(train_features.shape, val_features.shape, test_features.shape)
    #
    # pretrained_embedding_matrix = generate_embedding_matrix(tokenizer)
    #
    # train_driver([train_features, train_labels],
    #              [val_features, val_labels],
    #              pretrained_embedding_matrix)

    print('Elapsed time: {}'.format(time.time() - start_time))
