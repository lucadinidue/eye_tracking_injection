import sys
import os
sys.path.append(os.path.abspath('.'))

from modules.modeling.custom_modeling_roberta import RobertaForMultiTaskTokenClassification
from modules.modeling.custom_data_collator import DataCollatorForMultiTaskTokenClassification
from datasets import Dataset, load_dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from tqdm import tqdm
import pandas as pd
import argparse
import torch

TASKS = ['firstfix_dur','dur','firstrun_nfix','nfix','firstrun_dur']


def load_complexity_dataset(src_path:str) -> Dataset:
    df = pd.read_csv(src_path)
    annotators_columns = [col for col in df.columns if col.startswith('judgement')]
    df['label'] = df[annotators_columns].mean(axis=1)
    df = df[['SENTENCE', 'label']]
    df = df.rename(columns={'SENTENCE': 'text'})
    return Dataset.from_pandas(df)


def add_dummy_eye_gaze_features(example):
    for task in TASKS:
        example[f'label_{task}'] = [0]*len(example['input_ids'])
    return example


def predict_silver_labels(model, dataloader, output_path):
    model.eval()
    dataset_features = {task:[] for task in TASKS}
    with torch.no_grad():
        for el in tqdm(dataloader):
            logits = model(**el)['logits']
            for task in TASKS:
                dataset_features[task].append(logits[task].squeeze().tolist())
    
    features_df = pd.DataFrame.from_dict(dataset_features)
    features_df.to_pickle(output_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-u', '--user_id', dest='user_id', type=int)
    parser.add_argument('-t', '--downstream_task', dest='downstream_task', type=str, choices=['complexity', 'sentiment'])
    args = parser.parse_args()

    model_path = f'models/eye_gaze_finetuning/roberta-base_pp{args.user_id}'
    train_output_path = f'data/silver_labels/{args.downstream_task}/train_{args.user_id}.pkl'
    test_output_path = f'data/silver_labels/{args.downstream_task}/test_{args.user_id}.pkl'


    if args.downstream_task == 'complexity':
        train_path = 'data/complexity/complexity_ds_en_train.csv'
        test_path = 'data/complexity/complexity_ds_en_test.csv'
        train_dataset = load_complexity_dataset(train_path)
        test_dataset = load_complexity_dataset(test_path)
    else:
        dataset = load_dataset('sst2')
        dataset = dataset.rename_column("sentence", "text")
        train_dataset = dataset['train']
        test_dataset = dataset['validation']
    
    model = RobertaForMultiTaskTokenClassification.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained('FacebookAI/roberta-base', add_prefix_space=True)

    def preprocess_function(examples):
        return tokenizer(examples['text'], padding=False, truncation=True)

    tokenized_train_dataset = train_dataset.map(preprocess_function, remove_columns=['text'], batched=True)
    tokenized_test_dataset = test_dataset.map(preprocess_function, remove_columns=['text'], batched=True)

    data_collator = DataCollatorForMultiTaskTokenClassification(tokenizer=tokenizer)
    tokenized_train_dataset = tokenized_train_dataset.map(add_dummy_eye_gaze_features)
    tokenized_test_dataset = tokenized_test_dataset.map(add_dummy_eye_gaze_features)

    train_dataloader = DataLoader(tokenized_train_dataset, shuffle=False, collate_fn=data_collator, batch_size=1)
    test_dataloader = DataLoader(tokenized_test_dataset, shuffle=False, collate_fn=data_collator, batch_size=1)

    predict_silver_labels(model, train_dataloader, train_output_path)
    predict_silver_labels(model, test_dataloader, test_output_path)    



if __name__ == '__main__':
    main()