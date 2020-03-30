import os
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def score_roc_auc(target_csv, predicted_csv):
    train_df = pd.read_csv(target_csv).sort_values(by='id')
    compare_df = pd.read_csv(predicted_csv).sort_values(by='id')
    assert (train_df['id'].values == compare_df['id'].values).all()

    return roc_auc_score(train_df['toxic'].values,
                         compare_df['toxic'].values)


def generate_pseudo_labels(list_csvs,
                           text_csv,
                           text_col,
                           output_csv):
    base_df = pd.read_csv(text_csv)[['id', text_col]]
    base_df['toxic'] = np.mean(np.stack([pd.read_csv(x)['toxic'].values for x in list_csvs]), 0)
    base_df.to_csv(output_csv, index=False)


if __name__ == '__main__':
    generate_pseudo_labels(['data/test/bert-base-multilingual-cased.csv',
                            'data/test/bert-base-uncased.csv',
                            'data/test/bart-large.csv'],
                           'data/test_en.csv',
                           'content_en',
                           'data/pseudo_test.csv')
