"""
Generates the training data CSV file required for model training (either Transformer or FastText)
- Can generate data for monolingual or multilingual (incl. subset of the 6 test languages) models
- Saves training data to $TRAIN_DATA_DIR/curr_run_train.csv
- Saves validation data to $TRAIN_DATA_DIR/curr_run_val.csv
- Saves test data to $TRAIN_DATA_DIR/curr_run_test.csv (to predict against)
"""
import json
import os
import pandas as pd

LANG_LIST = ['es']  # list of test set language ISOs to create data for
SAMPLE_FRAC = 0.5  # Proportion of 2018 data (after filtering for LANG_LIST languages) to sub-sample for training

if __name__ == '__main__':
    with open('SETTINGS.json') as f:
        settings_dict = json.load(f)

    # Generate and save validation samples
    language_val = pd.read_csv(settings_dict['VALIDATION_PATH'])
    language_val = language_val[language_val['lang'].isin(LANG_LIST)].reset_index(drop=True)
    language_val.to_csv(os.path.join(settings_dict['TRAIN_DATA_DIR'], 'curr_run_val.csv'),
                        index=False)

    # Generate and save test samples
    test_df = pd.read_csv(settings_dict['PSEUDO_LABELS_PATH'])
    language_df = test_df[test_df.lang.isin(LANG_LIST)]
    language_df.columns = ['id', 'comment_text', 'lang', 'toxic']
    language_df.to_csv(os.path.join(settings_dict['TRAIN_DATA_DIR'], 'curr_run_test.csv'),
                       index=False)

    # Generate and save train samples
    translated_toxic = pd.read_csv(settings_dict['TRAIN_2018_PATH'])
    translated_toxic = translated_toxic[translated_toxic['lang'].isin(LANG_LIST)] \
        .sample(frac=SAMPLE_FRAC)
    translated_toxic = translated_toxic[['id', 'comment_text', 'lang', 'toxic']]
    pd.concat([language_df, translated_toxic]) \
        .reset_index(drop=True).to_csv(os.path.join(settings_dict['TRAIN_DATA_DIR'], 'curr_run_train.csv'),
                                       index=False)
