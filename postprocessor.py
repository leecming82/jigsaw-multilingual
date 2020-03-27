import pandas as pd
from sklearn.metrics import roc_auc_score


def score_roc_auc(target_csv, predicted_csv):
    train_df = pd.read_csv(target_csv).sort_values(by='id')
    compare_df = pd.read_csv(predicted_csv).sort_values(by='id')
    assert (train_df['id'].values == compare_df['id'].values).all()

    return roc_auc_score(train_df['toxic'].values,
                         compare_df['toxic'].values)


if __name__ == '__main__':
    print(score_roc_auc('data/validation.csv',
                        'data/submission.csv'))
