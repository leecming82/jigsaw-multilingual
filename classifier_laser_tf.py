"""
Classifier using LASER embeddings connected to a Dense layer
"""
import time
import pandas as pd
import numpy as np
from multiprocessing import Process, Queue
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.layers as layers
from tensorflow.keras import Model
from sklearn.metrics import roc_auc_score
from preprocessor import get_id_text_label_from_csv, get_id_text_from_test_csv
from laserembeddings import Laser

RUN_NAME = '9545es_laser'
USE_LANG = 'es'
TRAIN_CSV_PATH = 'data/es_all.csv'
TRAIN_SAMPLE_FRAC = 1.
TEST_CSV_PATH = 'data/es_test.csv'
VAL_CSV_PATH = 'data/validation_en.csv'
NUM_OUTPUTS = 1  # Number of targets
NUM_EPOCHS = 4
BATCH_SIZE = 32
HIDDEN_UNITS = 128

# gpus = tf.config.experimental.list_physical_devices('GPU')
# for gpu in gpus:
#     tf.config.experimental.set_memory_growth(gpu, True)


def build_classifier_model():
    input = layers.Input(shape=(1024,), dtype=np.float32)
    dense = layers.Dense(HIDDEN_UNITS, activation='relu')(input)
    prob = layers.Dense(NUM_OUTPUTS, activation='sigmoid')(dense)

    bigru_model = Model(input, prob)
    bigru_model.summary()
    print('generated laser model...')
    return bigru_model


def train_driver(train_tuple,
                 val_tuple,
                 test_tuple):
    train_laser_features, train_labels = train_tuple
    val_laser_features, val_labels = val_tuple
    test_laser_features, test_ids = test_tuple

    classifier = build_classifier_model()
    opt = Adam()
    opt = tf.keras.mixed_precision.experimental.LossScaleOptimizer(opt, 'dynamic')
    classifier.compile(optimizer=opt, loss='binary_crossentropy')

    for curr_epoch in range(NUM_EPOCHS):
        classifier.fit(train_laser_features, train_labels,
                       batch_size=BATCH_SIZE,
                       epochs=1,
                       verbose=1)

        if len(val_labels):
            val_preds = classifier.predict(val_laser_features)
            val_roc_auc_score = roc_auc_score(val_labels, val_preds)
            print(val_roc_auc_score)

        test_preds = classifier.predict(test_laser_features).squeeze()
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

    print('Generating LASER embeddings...')
    # Use a process and queue to generate embeddings so can ensure GPU memory is freed up after features
    # are generated
    laser_queue = Queue()
    def generate_laser_features():
        laser = Laser()
        train_laser_features = laser.embed_sentences(train_strings, lang=USE_LANG)
        val_laser_features = laser.embed_sentences(val_strings, lang=USE_LANG)
        test_laser_features = laser.embed_sentences(test_strings, lang=USE_LANG)
        laser_queue.put((train_laser_features, val_laser_features, test_laser_features))

    p1 = Process(target=generate_laser_features)
    p1.start()
    train_laser_features, val_laser_features, test_laser_features = laser_queue.get()
    p1.join()
    print(train_laser_features.shape, val_laser_features.shape, test_laser_features.shape)
    print(train_laser_features[0])

    train_driver([train_laser_features, train_labels],
                 [val_laser_features, val_labels],
                 [test_laser_features, test_ids])

    print('Elapsed time: {}'.format(time.time() - start_time))
