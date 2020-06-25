"""
Averages the temporal ensembled prediction CSVs created by a training run,
blends them with the previous ensembled predictions,
saves to a single CSV ready for submission to the LB
- Averages epoch predictions and saves to $TRAIN_DATA_DIR/curr_run_preds.csv
- Blends with previous ensemble and saves to $TRAIN_DATA_DIR/curr_run_submission.csv
"""
import json
import os
import pandas as pd
from postprocessor import ensemble_simple_avg_csv

# blend weight of the previous ensembled predictions (i.e., current preds will have 1-ENSEMBLE_WEIGHT weight)
ENSEMBLE_WEIGHT = 0.5

if __name__ == '__main__':
    with open('SETTINGS.json') as f:
        settings_dict = json.load(f)

    x = sorted([os.path.join(settings_dict['PREDICTION_DIR'], x) for x in os.listdir(settings_dict['PREDICTION_DIR'])])

    # Average-ensemble current run's predictions and save
    ensemble_simple_avg_csv(x, output_path=os.path.join(settings_dict['TRAIN_DATA_DIR'], 'curr_run_preds.csv'))
    preds_df = pd.read_csv(os.path.join(settings_dict['TRAIN_DATA_DIR'], 'curr_run_preds.csv'))

    # Load previous ensembled predictions
    test_df = pd.read_csv(settings_dict['PSEUDO_LABELS_PATH'])

    # Blend and save
    test_df.loc[preds_df.id.values, 'toxic'] = ENSEMBLE_WEIGHT * test_df.loc[preds_df.id.values, 'toxic'].values + \
                                               (1-ENSEMBLE_WEIGHT) * preds_df['toxic'].values
    test_df[['id', 'toxic']].to_csv(os.path.join(settings_dict['TRAIN_DATA_DIR'], 'curr_run_submission.csv'),
                                    index=False)
