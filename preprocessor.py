import pandas as pd
from sklearn.model_selection import KFold

SEED = 1337
NUM_FOLDS = 4


def generate_train_kfolds_indices(input_df):
    """
    Seeded kfolds cross validation indices using just a range(len) call
    :return: (training index, validation index)-tuple list
    """
    seeded_kf = KFold(n_splits=NUM_FOLDS, random_state=SEED, shuffle=True)
    return [(train_index, val_index) for train_index, val_index in
            seeded_kf.split(range(len(input_df)))]


def get_id_text_label_from_csv(csv_path):
    """
    Load training data
    :param csv_path: path of csv with 'id' 'comment_text', 'toxic' columns present
    :return:
    """
    raw_pdf = pd.read_csv(csv_path)
    return raw_pdf['id'].values, list(raw_pdf['comment_text'].values), raw_pdf['toxic'].values


def get_id_text_from_test_csv(csv_path):
    """
    Load training data
    :param csv_path: path of csv with 'id' 'comment_text' columns present
    :return:
    """
    raw_pdf = pd.read_csv(csv_path)
    return raw_pdf['id'].values, list(raw_pdf['content'].values)


if __name__ == '__main__':
    ids, comments, labels = get_id_text_label_from_csv('data/validation.csv')
    print(ids)
    print(comments)
    print(labels)
