import pandas as pd
import random

TEST_FRACTION = 0.2

def main():
    random.seed(42)

    src_path = 'data/complexity/complexity_ds_en.csv'
    train_path = 'data/complexity/complexity_ds_en_train.csv'
    test_path = 'data/complexity/complexity_ds_en_test.csv'

    df = pd.read_csv(src_path)
    
    num_test_ids = int(len(df)*TEST_FRACTION)
    test_ids = random.sample(df['ID'].unique().tolist(), num_test_ids)
    
    test_df = df[df['ID'].isin(test_ids)]
    train_df = df[~df['ID'].isin(test_ids)]

    train_df.to_csv(train_path)
    test_df.to_csv(test_path)

if __name__ == '__main__':
    main()