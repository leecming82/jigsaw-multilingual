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
    base_df.reset_index().to_csv('data/it_{}.csv'.format(len(list_csv)), index=False)


def ensemble_power_avg_csv(list_csv, power):
    base_df = pd.read_csv(list_csv[0]).sort_values('id').set_index('id')
    for i in range(1, len(list_csv)):
        base_df['toxic'] += pd.read_csv(list_csv[i]).sort_values('id').set_index('id')['toxic'].values**power
    base_df['toxic'] /= len(list_csv)
    base_df = base_df.reset_index()
    base_df[['id', 'toxic']].to_csv('data/power_ensemble_{}.csv'.format(len(list_csv)), index=False)

if __name__ == '__main__':
    # x = sorted([os.path.join('data/outputs/test/', x) for x in os.listdir('data/outputs/test') if len([y for y in ['_0.csv', '_1.csv', '_2.csv'] if y in x]) == 0])
    # ensemble_simple_avg_csv(x)

    ensemble_simple_avg_csv(['data/outputs/test/bert-base-italian-xxl-cased_0.csv',
                             'data/outputs/test/bert-base-italian-xxl-cased_1.csv',
                             'data/outputs/test/bert-base-italian-xxl-cased_2.csv'])
