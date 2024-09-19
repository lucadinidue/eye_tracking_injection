import pandas as pd
import random
import os


SEED = 12345
random.seed(SEED)


COLUMNS_MAPPING = {
        "PP_NR": "uniform_id",
        "PART": "trialid",
        "TRIAL": "sentnum",
        "WORD_ID": "ianum",
        "WORD": "ia",
        "PP_NR": "uniform_id",
        "WORD_FIRST_FIXATION_DURATION": "firstfix_dur",
        "WORD_FIRST_RUN_DURATION": "firstrun_dur",
        "WORD_TOTAL_READING_TIME": "dur",
        "WORD_FIRST_RUN_FIXATION_COUNT": "firstrun_nfix",
        "WORD_FIXATION_COUNT": "nfix"
    }


def preprocess_dataset_features(df: pd.DataFrame) -> pd.DataFrame:
    gaze_features = ["WORD_FIRST_FIXATION_DURATION", "WORD_FIRST_RUN_END_TIME", "WORD_FIRST_RUN_START_TIME", "WORD_TOTAL_READING_TIME", "WORD_FIRST_RUN_FIXATION_COUNT", "WORD_FIXATION_COUNT"]
    other_features = ["PP_NR", "PART", "TRIAL", "WORD_ID", "WORD"] # trial is the paragraph readed
    df = df[other_features + gaze_features]

    # Reasonably the features with "." are related to skipped words
    # We can substitute them with 0 value.

    for gaze_feat in gaze_features:
        df.loc[df[gaze_feat] == ".", gaze_feat] = 0

    df[gaze_features] = df[gaze_features].astype(float)
    df["WORD_FIRST_RUN_DURATION"] = df["WORD_FIRST_RUN_END_TIME"] - df["WORD_FIRST_RUN_START_TIME"]

    del df["WORD_FIRST_RUN_END_TIME"]
    del df["WORD_FIRST_RUN_START_TIME"]

    df.dropna()

    return df


def process_user_df(user_id, user_df, test_trial_ids, out_dir):
    user_df["WORD_ID"] = user_df.index
    user_df["WORD_ID"] = user_df["WORD_ID"].astype(float)
    user_df["PART"] = user_df["PART"].astype(float)
    user_df["TRIAL"] = user_df["TRIAL"].astype(float)
    user_df.rename(columns=COLUMNS_MAPPING, inplace=True)

    test_ids = user_df.sentnum.isin(test_trial_ids)
    train_ids = ~test_ids

    train_df = user_df[train_ids]
    test_df = user_df[test_ids]

    train_df.to_csv(os.path.join(out_dir, f'{user_id}_dataset_train.csv'))
    test_df.to_csv(os.path.join(out_dir, f'{user_id}_dataset_test.csv'))

def main():
    dataset_path = 'data/geco/geco_monolingual.csv'
    out_dir = 'data/geco/dataset/'

    df = pd.read_csv(dataset_path, index_col=0)
    df = preprocess_dataset_features(df)

    user_ids = df['PP_NR'].unique().tolist()
    trial_ids = df['TRIAL'].unique().tolist()
    test_trial_ids = random.sample(trial_ids, 20)

    print('TEST IDS:', test_trial_ids)

    for user_id in user_ids:
        user_df = df[df.PP_NR == user_id].reset_index(drop=True)
        if len(user_df) == 56410:
            process_user_df(user_id, user_df, test_trial_ids, out_dir)
            
           
if __name__ == "__main__":
    main()