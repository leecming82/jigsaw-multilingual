import numpy as np
import pandas as pd
from functools import lru_cache
from sklearn.model_selection import KFold
from scipy.stats import truncnorm

SEED = 1337
NUM_FOLDS = 4
TOXIC_TARGET_COLS = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
LANG_MAPPING = {lang: np.identity(7)[i] for i, lang in enumerate(['en', 'tr', 'pt', 'ru', 'fr', 'it', 'es'])}


def generate_train_kfolds_indices(input_df):
    """
    Seeded kfolds cross validation indices using just a range(len) call
    :return: (training index, validation index)-tuple list
    """
    seeded_kf = KFold(n_splits=NUM_FOLDS, random_state=SEED, shuffle=True)
    return [(train_index, val_index) for train_index, val_index in
            seeded_kf.split(range(len(input_df)))]


def get_id_text_label_from_csv(csv_path, text_col='comment_text',
                               sample_frac=1.,
                               add_label=None):
    """
    Load training data
    :param csv_path: path of csv with 'id' 'comment_text', 'toxic' columns present
    :param text_col: specify the text col name (for translations)
    :return:
    """
    raw_df = pd.read_csv(csv_path)
    if sample_frac < 1:
        raw_df = raw_df.sample(frac=sample_frac)
    if add_label is None:
        return raw_df['id'].values, list(raw_df[text_col].values), raw_df['toxic'].values
    else:
        return raw_df['id'].values, \
               list(raw_df[text_col].values), \
               raw_df['toxic'].values, np.full(raw_df.shape[0], add_label)


def get_id_text_from_test_csv(csv_path, text_col):
    """
    Load test data
    :param csv_path: path of csv with 'id' 'comment_text' columns present
    :param text_col: column w/ test
    :return:
    """
    raw_pdf = pd.read_csv(csv_path)
    return raw_pdf['id'].values, list(raw_pdf[text_col].values)


def get_id_text_toxic_labels_from_csv(csv_path, text_col='comment_text', sample_frac=1.):
    """
    Load training data w/ all 6 toxic targets e.g., toxic, severe_toxic etc.
    :param csv_path: path of csv with 'id' 'comment_text', 'toxic' columns present
    :return:
    """
    raw_df = pd.read_csv(csv_path)
    if sample_frac < 1:
        raw_df = raw_df.sample(frac=sample_frac, random_state=SEED)
    return raw_df['id'].values, list(raw_df[text_col].values), raw_df[TOXIC_TARGET_COLS].values


def get_id_text_distill_label_from_csv(train_path, distill_path, text_col='comment_text', sample_frac=1.):
    """
    Load training data together with distillation labels
    :param train_path: path with original labels
    :param distill_path: path distill labels
    :param text_col: specify the text col name (for translations)
    :return:
    """
    raw_df = pd.read_csv(train_path)
    if sample_frac < 1:
        raw_df = raw_df.sample(frac=sample_frac, random_state=SEED)
    distill_df = pd.read_csv(distill_path).set_index('id')
    distill_df = distill_df.loc[raw_df['id']]
    return (raw_df['id'].values,
            list(raw_df[text_col].values),
            distill_df['toxic'].values)


def get_id_text_label_from_csvs(list_csv_path, sample_frac=.1):
    """
    Load training data from multiple csvs
    :param csv_path: list of csv with 'id' 'comment_text', 'toxic' columns present
    :return:
    """
    raw_df = pd.concat([pd.read_csv(csv_path)[['id', 'comment_text', 'toxic']] for csv_path in list_csv_path])
    if sample_frac < 1:
        raw_df = raw_df.sample(frac=sample_frac, random_state=SEED)
    assert raw_df['id'].nunique() == raw_df.shape[0]
    return raw_df['id'].values, list(raw_df['comment_text'].values), raw_df['toxic'].values


@lru_cache(maxsize=None)
def generate_target_dist(mean, num_bins, low, high):
    """
    Generate discretized truncated norm prob distribution centered around mean
    :param mean: center of truncated norm
    :param num_bins: number of bins
    :param low: low end of truncated range
    :param high: top end of truncated range
    :return: (support, probabilities for support) tuple
    """
    radius = 0.5 * (high - low) / num_bins

    def trunc_norm_prob(center):
        """ get probability mass """
        return (truncnorm.cdf(center + radius,
                              a=(low - mean) / radius,
                              b=(high - mean) / radius,
                              loc=mean, scale=radius) -
                truncnorm.cdf(center - radius,
                              a=(low - mean) / radius,
                              b=(high - mean) / radius,
                              loc=mean, scale=radius))

    supports = np.array([x * (2 * radius) + radius + low for x in range(num_bins)])
    probs = np.array([trunc_norm_prob(support) for support in supports])
    return supports, probs


if __name__ == '__main__':
    # ids, texts, targets = get_id_text_toxic_labels_from_csv('./data/toxic_2018/combined.csv')
    print(LANG_MAPPING['it'])