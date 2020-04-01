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


def ensemble_simple_avg_csv(list_csv):
    base_df = pd.read_csv(list_csv[0]).sort_values('id').set_index('id')
    for i in range(1, len(list_csv)):
        base_df['toxic'] += pd.read_csv(list_csv[i]).sort_values('id').set_index('id')['toxic']
    base_df['toxic'] /= len(list_csv)
    base_df.reset_index().to_csv('data/ensemble_{}.csv'.format(len(list_csv)), index=False)


if __name__ == '__main__':
    ensemble_simple_avg_csv(['data/submission_train_val_2_avg.csv',
                             'data/submission_train_val_3_avg.csv',
                             'data/submission_train_val_4_avg.csv',
                             'data/submission_train_val_5_avg.csv'])
