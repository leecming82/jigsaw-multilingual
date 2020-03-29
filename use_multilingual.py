"""
Use pretrained Universal Sentence Encoder Multi-Lingual embeddings w/ an MLP to classify
"""
import os
import time
import numpy as np
import tensorflow as tf
import tensorflow_text
import tensorflow.keras.layers as layers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Model
from sklearn.metrics import roc_auc_score
from preprocessor import get_id_text_label_from_csv, get_id_text_distill_label_from_csv

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

SAVE_MODEL = False
USE_DISTILL = True
USE_MODEL_PATH = './models/use_ml/large'
PRETRAINED_MODEL = 'distilbert-base-uncased'
# TRAIN_CSV_PATH = 'data/jigsaw-toxic-comment-train.csv'
# DISTIL_CSV_PATH = None
TRAIN_CSV_PATH = 'data/toxic_2018/train.csv'
DISTIL_CSV_PATH = 'data/toxic_2018/ensemble_3.csv'
VAL_CSV_PATH = 'data/validation_en.csv'
OUTPUT_DIR = 'models/'
NUM_GPUS = 2  # Set to 1 if using AMP (doesn't seem to play nice with 1080 Ti)
MAX_CORES = 24  # limit MP calls to use this # cores at most
NUM_HIDDEN = 256
OUTPUT_DIM = 1
MAX_SEQ_LEN = 200  # max sequence length for input strings: gets padded/truncated
NUM_EPOCHS = 5
BATCH_SIZE = 64

def build_classifier_model():
    input = layers.Input(shape=(512,), dtype=tf.float32)
    hidden = layers.Dense(NUM_HIDDEN, activation='relu')(input)
    prob = layers.Dense(OUTPUT_DIM, activation='sigmoid')(hidden)

    model = Model(input, prob)
    model.summary()
    return model


def main_driver(train_tuple, val_raw_tuple, val_translated_tuple):
    [train_features, train_labels, _] = train_tuple
    [val_raw_features, val_labels, _] = val_raw_tuple
    [val_translated_features, _, _] = val_translated_tuple

    classifier = build_classifier_model()
    opt = Adam()

    opt.learning_rate = 1e-3
    classifier.compile(optimizer=opt,
                       loss='binary_crossentropy')

    for _ in range(NUM_EPOCHS):
        classifier.fit(train_features, train_labels,
                       batch_size=BATCH_SIZE,
                       epochs=1,
                       verbose=0)
        val_raw_preds = classifier.predict(val_raw_features)
        val_translated_preds = classifier.predict(val_translated_features)
        val_raw_score = roc_auc_score(val_labels, val_raw_preds)
        val_translated_score = roc_auc_score(val_labels, val_translated_preds)
        print('Raw AUC: {}, Translated AUC: {}'.format(val_raw_score, val_translated_score))


if __name__ == '__main__':
    start_time = time.time()

    embed = tf.saved_model.load(USE_MODEL_PATH)

    if USE_DISTILL:
        train_ids, train_strings, train_labels = get_id_text_distill_label_from_csv(TRAIN_CSV_PATH, DISTIL_CSV_PATH)
    else:
        train_ids, train_strings, train_labels = get_id_text_label_from_csv(TRAIN_CSV_PATH)
    val_ids, val_raw_strings, val_labels = get_id_text_label_from_csv(VAL_CSV_PATH, text_col='comment_text')
    _, val_translated_strings, _ = get_id_text_label_from_csv(VAL_CSV_PATH, text_col='comment_text_en')

    train_features = np.concatenate([embed(train_strings[x:x + 128]).numpy()
                                     for x in range(0, len(train_strings), 128)])
    val_raw_features = np.concatenate([embed(val_raw_strings[x:x + 128]).numpy()
                                       for x in range(0, len(val_raw_strings), 128)])
    val_translated_features = np.concatenate([embed(val_translated_strings[x:x + 128]).numpy()
                                              for x in range(0, len(val_translated_strings), 128)])
    print(train_features.shape, val_raw_features.shape, val_translated_features.shape)

    main_driver([train_features, train_labels, train_ids],
                [val_raw_features, val_labels, val_ids],
                [val_translated_features, val_labels, val_ids])

    print('ELapsed time: {}'.format(time.time() - start_time))
