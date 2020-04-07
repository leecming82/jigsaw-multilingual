"""
Baseline TF2 classifier for Jigsaw Multilingual
- Assumes two separate train and val sets (i.e., no need for k-folds)
"""
import os
import time
from functools import partial
import multiprocessing as mp
import numpy as np
from transformers import AutoTokenizer, TFAutoModel, AutoConfig
from sklearn.metrics import roc_auc_score
from preprocessor import get_id_text_label_from_csv
import tensorflow as tf
from tensorflow.keras.optimizers import Adam

SAVE_MODEL = True
USE_AMP = True
USE_XLA = False
USE_PSEUDO = False  # Add pseudo labels to training dataset
PRETRAINED_MODEL = 'jplu/tf-xlm-roberta-large'
TRAIN_SAMPLE_FRAC = .05  # what % of training data to use
TRAIN_CSV_PATH = 'data/translated_2018/combined.csv'
VAL_CSV_PATH = 'data/validation_en.csv'
PSEUDO_CSV_PATH = 'data/test/test9337.csv'
OUTPUT_DIR = 'models/translated_xlmr_msdo'
NUM_GPUS = 2
MAX_CORES = 24  # limit MP calls to use this # cores at most
NUM_OUTPUTS = 1
MAX_SEQ_LEN = 200  # max sequence length for input strings: gets padded/truncated
NUM_EPOCHS = 6
BATCH_SIZE = 32
LR_START = 1e-3
LR_FINETUNE = 1e-5

# os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'

tf.autograph.set_verbosity(0)
tf.config.optimizer.set_jit(USE_XLA)
tf.config.optimizer.set_experimental_options({"auto_mixed_precision": USE_AMP})
# tf.config.experimental.set_memory_growth(tf.config.experimental.list_physical_devices('GPU')[0], True)


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
    pretrained_base = TFAutoModel.from_pretrained(PRETRAINED_MODEL, config=pretrained_config)
    for layer in pretrained_base.layers:
        layer.trainable = False

    hidden_layer = pretrained_base(input)[0]

    base_output = layers.Conv1D(NUM_OUTPUTS, 1)(hidden_layer)
    base_output = layers.GlobalMaxPooling1D()(base_output)
    prob = layers.Activation('sigmoid')(base_output)
    model = Model(input, prob)

    model.summary()

    return model


def main_driver(train_tuple, val_raw_tuple):
    train_features, train_labels, _ = train_tuple
    val_raw_features, val_labels, _ = val_raw_tuple

    mirrored_strategy = tf.distribute.MirroredStrategy()
    with mirrored_strategy.scope():
        classifier = build_classifier_model()
        opt = Adam(lr=LR_START)
        if USE_AMP:
            opt = tf.keras.mixed_precision.experimental.LossScaleOptimizer(opt, 'dynamic')
        classifier.compile(optimizer=opt, loss='binary_crossentropy')

    epoch_eval_score = []

    for curr_epoch in range(NUM_EPOCHS):
        if curr_epoch == 1:
            with mirrored_strategy.scope():
                opt.learning_rate = LR_FINETUNE
                for layer in classifier.layers[1].layers:
                    layer.trainable = True
                classifier.compile(optimizer=opt,
                                   loss='binary_crossentropy')

        classifier.fit(train_features, train_labels,
                       batch_size=BATCH_SIZE,
                       epochs=1,
                       verbose=1)
        val_preds = classifier.predict(val_raw_features)
        val_roc_auc_score = roc_auc_score(val_labels, val_preds)
        epoch_eval_score.append(val_roc_auc_score)
        print('Epoch {} eval score: {}'.format(curr_epoch, val_roc_auc_score))


if __name__ == '__main__':
    start_time = time.time()

    # Load train, validation, and pseudo-label data
    train_ids, train_strings, train_labels = get_id_text_label_from_csv(TRAIN_CSV_PATH,
                                                                        text_col='comment_text',
                                                                        sample_frac=TRAIN_SAMPLE_FRAC)

    val_ids, val_raw_strings, val_labels = get_id_text_label_from_csv(VAL_CSV_PATH, text_col='comment_text')
    # _, val_translated_strings, _ = get_id_text_label_from_csv(VAL_CSV_PATH, text_col='comment_text_en')

    pseudo_ids, pseudo_strings, pseudo_labels = get_id_text_label_from_csv(PSEUDO_CSV_PATH, text_col='content')

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
        train_features = np.array(p.map(encode_partial, train_strings))
        val_raw_features = np.array(p.map(encode_partial, val_raw_strings))
        # val_translated_features = np.array(p.map(encode_partial, val_translated_strings))
        pseudo_features = None
        if USE_PSEUDO:
            pseudo_features = np.array(p.map(encode_partial, pseudo_strings))

    # train_features = np.concatenate([train_features, val_raw_features])
    # train_labels = np.concatenate([train_labels, val_labels])
    # train_ids = np.concatenate([train_ids, val_ids])
    if USE_PSEUDO:
        train_features = np.concatenate([train_features, pseudo_features])
        train_labels = np.concatenate([train_labels, pseudo_labels])
        train_ids = np.concatenate([train_ids, pseudo_ids])

    print('Train size: {}, val size: {}, pseudo size: {}'.format(len(train_ids), len(val_ids), len(pseudo_ids)))
    print('Train positives: {}, train negatives: {}'.format(train_labels[train_labels == 1].shape,
                                                            train_labels[train_labels == 0].shape))

    main_driver([train_features, train_labels, train_ids],
                [val_raw_features, val_labels, val_ids])

    print('Elapsed time: {}'.format(time.time() - start_time))
