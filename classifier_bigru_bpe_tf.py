"""
Classifier using BiGRU w/ pretrained bpe embeddings
"""
import time
import multiprocessing as mp
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.layers as layers
from tensorflow.keras import Model
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import roc_auc_score
from preprocessor import get_id_text_label_from_csv, get_id_text_from_test_csv
from bpemb import BPEmb

RUN_NAME = '9544fr_bpe'
USE_LANG = 'fr'
VOCAB_SIZE = 100000
EMBEDDING_DIMS = 300
BPE_MODEL = BPEmb(lang=USE_LANG, vs=VOCAB_SIZE, dim=EMBEDDING_DIMS)
TRAIN_CSV_PATH = 'data/fr_all.csv'
TRAIN_SAMPLE_FRAC = 1.
TEST_CSV_PATH = 'data/fr_test.csv'
VAL_CSV_PATH = 'data/validation_en.csv'
NUM_OUTPUTS = 1  # Number of targets
MAX_SEQ_LEN = 200  # max sequence length for input strings: gets padded/truncated
NUM_EPOCHS = 4
BATCH_SIZE = 32
HIDDEN_UNITS = 128
MAX_CORES = 24


def build_classifier_model():
    input = layers.Input(shape=(MAX_SEQ_LEN,), dtype=np.int32)

    embedding_layer = layers.Embedding(VOCAB_SIZE,
                                       EMBEDDING_DIMS,
                                       weights=[BPE_MODEL.vectors],
                                       trainable=False)
    embedded_input = embedding_layer(input)
    gru_output = layers.Bidirectional(layers.GRU(HIDDEN_UNITS,
                                                 return_sequences=True))(embedded_input)
    gru_output = layers.Bidirectional(layers.GRU(HIDDEN_UNITS))(gru_output)
    prob = layers.Dense(NUM_OUTPUTS, activation='sigmoid')(gru_output)

    bigru_model = Model(input, prob)
    print('generated bigru model...')
    return bigru_model


def train_driver(train_tuple,
                 val_tuple,
                 test_tuple):
    train_features, train_labels = train_tuple
    val_features, val_labels = val_tuple
    test_features, test_ids = test_tuple

    classifier = build_classifier_model()
    opt = Adam()
    opt = tf.keras.mixed_precision.experimental.LossScaleOptimizer(opt, 'dynamic')
    classifier.compile(optimizer=opt, loss='binary_crossentropy')

    for curr_epoch in range(NUM_EPOCHS):
        classifier.fit(train_features, train_labels,
                       batch_size=BATCH_SIZE,
                       epochs=1,
                       verbose=1)

        if len(val_labels):
            val_preds = classifier.predict(val_features)
            val_roc_auc_score = roc_auc_score(val_labels, val_preds)
            print(val_roc_auc_score)

        test_preds = classifier.predict(test_features).squeeze()
        pd.DataFrame({'id': test_ids, 'toxic': test_preds}) \
            .to_csv('data/outputs/test/{}_{}.csv'.format(RUN_NAME,
                                                         curr_epoch),
                    index=False)


if __name__ == '__main__':
    start_time = time.time()
    # Load train, validation, and pseudo-label data
    train_ids, train_strings, train_labels = get_id_text_label_from_csv(TRAIN_CSV_PATH,
                                                                        text_col='comment_text',
                                                                        sample_frac=TRAIN_SAMPLE_FRAC)
    val_ids, val_strings, val_labels = get_id_text_label_from_csv(VAL_CSV_PATH,
                                                                  text_col='comment_text',
                                                                  lang=USE_LANG)
    test_ids, test_strings = get_id_text_from_test_csv(TEST_CSV_PATH, text_col='comment_text')

    print('Encoding raw strings into model-specific tokens')
    with mp.Pool(MAX_CORES) as p:
        train_features = list(p.map(BPE_MODEL.encode_ids, train_strings))
        val_features = list(p.map(BPE_MODEL.encode_ids, val_strings))
        test_features = list(p.map(BPE_MODEL.encode_ids, test_strings))

    train_features = pad_sequences(train_features, maxlen=MAX_SEQ_LEN)
    val_features = pad_sequences(val_features, maxlen=MAX_SEQ_LEN)
    test_features = pad_sequences(test_features, maxlen=MAX_SEQ_LEN)

    print(train_features.shape, val_features.shape, test_features.shape)

    train_driver([train_features, train_labels],
                 [val_features, val_labels],
                 [test_features, test_ids])

    print('Elapsed time: {}'.format(time.time() - start_time))
