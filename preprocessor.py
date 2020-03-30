import pandas as pd
from sklearn.model_selection import KFold

SEED = 1337
NUM_FOLDS = 4
TOXIC_TARGET_COLS = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']


def generate_train_kfolds_indices(input_df):
    """
    Seeded kfolds cross validation indices using just a range(len) call
    :return: (training index, validation index)-tuple list
    """
    seeded_kf = KFold(n_splits=NUM_FOLDS, random_state=SEED, shuffle=True)
    return [(train_index, val_index) for train_index, val_index in
            seeded_kf.split(range(len(input_df)))]


def get_id_text_label_from_csv(csv_path, text_col='comment_text', sample_frac=1.):
    """
    Load training data
    :param csv_path: path of csv with 'id' 'comment_text', 'toxic' columns present
    :param text_col: specify the text col name (for translations)
    :return:
    """
    raw_df = pd.read_csv(csv_path)
    if sample_frac < 1:
        raw_df = raw_df.sample(frac=sample_frac, random_state=SEED)
    return raw_df['id'].values, list(raw_df[text_col].values), raw_df['toxic'].values


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


if __name__ == '__main__':
    ids, comments, labels = get_id_text_label_from_csvs(['data/jigsaw-toxic-comment-train.csv',
                                                         'data/jigsaw-unintended-bias-train.csv'])
