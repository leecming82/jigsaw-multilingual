"""
Baseline TF2 classifier for Jigsaw Multilingual
"""
import os
import time
from functools import partial
import multiprocessing as mp
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, TFAutoModel, AutoConfig
from sklearn.metrics import roc_auc_score
from preprocessor import get_id_text_label_from_csv, get_id_text_from_test_csv
import tensorflow as tf
from tensorflow.keras.optimizers import Adam

RUN_NAME = '9545es_tf'
USE_AMP = True
USE_XLA = True
USE_PSEUDO = False  # Add pseudo labels to training dataset
FROM_PT = True  # Need to set a flag in TFAutoModel if the pretrained model is a torch model
USE_VAL_LANG = 'es'  # if set to ISO lang str (e.g., "es") - only pulls that language's validation samples
PRETRAINED_MODEL = 'mrm8488/distill-bert-base-spanish-wwm-cased-finetuned-spa-squad2-es'
TRAIN_SAMPLE_FRAC = 1.  # what % of training data to use
TRAIN_CSV_PATH = 'data/es_all.csv'
TEST_CSV_PATH = 'data/es_test.csv'
VAL_CSV_PATH = 'data/validation.csv'
MAX_CORES = 24  # limit MP calls to use this # cores at most
NUM_OUTPUTS = 1
MAX_SEQ_LEN = 200  # max sequence length for input strings: gets padded/truncated
NUM_EPOCHS = 4
BATCH_SIZE = 64
LR = 1e-5

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

tf.autograph.set_verbosity(0)
tf.config.optimizer.set_jit(USE_XLA)
tf.config.experimental.set_memory_growth(tf.config.experimental.list_physical_devices('GPU')[0], True)


def build_classifier_model():
    import tensorflow.keras.layers as layers
    from tensorflow.keras import Model
    """
    Bert base with a Linear layer plopped on top of it
    - connects the max pool of the last hidden layer with the FC
    """
    input = layers.Input(shape=(MAX_SEQ_LEN,), dtype=np.int32)
    pretrained_config = AutoConfig.from_pretrained(PRETRAINED_MODEL,
                                                   output_hidden_states=True)
    pretrained_base = TFAutoModel.from_pretrained(PRETRAINED_MODEL, config=pretrained_config, from_pt=FROM_PT)
    hidden_layer = pretrained_base(input)[0]

    # base_output = layers.Conv1D(NUM_OUTPUTS, 1)(hidden_layer)
    # base_output = layers.GlobalMaxPooling1D()(base_output)
    cls_token = hidden_layer[:, 0, :]
    prob = layers.Dense(1, activation='sigmoid')(cls_token)
    model = Model(input, prob)

    model.summary()

    return model


def main_driver(train_tuple, val_tuple, test_tuple):
    train_features, train_labels, _ = train_tuple
    val_features, val_labels, _ = val_tuple
    test_features, test_ids = test_tuple

    # mirrored_strategy = tf.distribute.MirroredStrategy()
    # with mirrored_strategy.scope():
    classifier = build_classifier_model()
    opt = Adam(lr=LR)
    if USE_AMP:
        opt = tf.train.experimental.enable_mixed_precision_graph_rewrite(opt)
    classifier.compile(optimizer=opt, loss='binary_crossentropy')

    epoch_eval_score = []
    for curr_epoch in range(NUM_EPOCHS):
        classifier.fit(train_features, train_labels,
                       batch_size=BATCH_SIZE,
                       epochs=1,
                       verbose=1)
        val_preds = classifier.predict(val_features)
        val_roc_auc_score = roc_auc_score(val_labels, val_preds)
        epoch_eval_score.append(val_roc_auc_score)
        print('Epoch {} eval score: {}'.format(curr_epoch, val_roc_auc_score))

        test_preds = classifier.predict(test_features).squeeze()
        pd.DataFrame({'id': test_ids, 'toxic': test_preds}).to_csv('data/outputs/test/{}_{}.csv'.format(RUN_NAME,
                                                                                                        curr_epoch),
                                                                   index=False)

    with np.printoptions(precision=4, suppress=True):
        print(np.array(epoch_eval_score))


if __name__ == '__main__':
    def cln(x):
        return ' '.join(x.split())


    start_time = time.time()
    print(RUN_NAME)

    # Load train, validation, and pseudo-label data
    train_ids, train_strings, train_labels = get_id_text_label_from_csv(TRAIN_CSV_PATH,
                                                                        text_col='comment_text',
                                                                        sample_frac=TRAIN_SAMPLE_FRAC)
    print(train_strings[0])
    train_strings = [cln(x) for x in train_strings]
    print(train_strings[0])

    val_ids, val_strings, val_labels = get_id_text_label_from_csv(VAL_CSV_PATH, text_col='comment_text',
                                                                  lang=USE_VAL_LANG)
    val_strings = [cln(x) for x in val_strings]

    test_ids, test_strings = get_id_text_from_test_csv(TEST_CSV_PATH, text_col='comment_text')
    test_strings = [cln(x) for x in test_strings]

    # use MP to batch encode the raw feature strings into Bert token IDs
    tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_MODEL)
    encode_partial = partial(tokenizer.encode,
                             max_length=MAX_SEQ_LEN,
                             pad_to_max_length=True,
                             add_special_tokens=True)
    print('Encoding raw strings into model-specific tokens')
    with mp.Pool(MAX_CORES) as p:
        train_features = np.array(p.map(encode_partial, train_strings))
        val_features = np.array(p.map(encode_partial, val_strings))
        test_features = np.array(p.map(encode_partial, test_strings))

    print('Train size: {}, val size: {}'.format(len(train_ids), len(val_ids)))
    print('Train positives: {}, train negatives: {}'.format(train_labels[train_labels == 1].shape,
                                                            train_labels[train_labels == 0].shape))

    main_driver([train_features, train_labels, train_ids],
                [val_features, val_labels, val_ids],
                [test_features, test_ids])

    print('Elapsed time: {}'.format(time.time() - start_time))
