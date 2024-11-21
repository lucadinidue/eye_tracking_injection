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

TASK_TO_KEYS = {
    "complexity": ("SENTENCE", None),
    "sentiment": ("sentence", None),
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}

def load_complexity_dataset(src_path:str) -> Dataset:
    df = pd.read_csv(src_path)
    annotators_columns = [col for col in df.columns if col.startswith('judgement')]
    df['label'] = df[annotators_columns].mean(axis=1)
    df = df[['SENTENCE', 'label']]
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


def prepare_dataset_and_predict(dataset, downstream_task, tokenizer, output_path, model, data_collator, batch_size=1):
    sentence1_key, sentence2_key = TASK_TO_KEYS[downstream_task]

    def preprocess_function(examples):
        args = ((examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key]))
        return tokenizer(*args, padding=True, truncation=True)
    
    cols_to_remove = list(TASK_TO_KEYS[downstream_task]) if TASK_TO_KEYS[downstream_task][1] is not None else [TASK_TO_KEYS[downstream_task][0]]
    cols_to_remove.append('idx') 
    tokenized_dataset = dataset.map(preprocess_function, remove_columns=cols_to_remove, batched=True)
    tokenized_dataset = tokenized_dataset.map(add_dummy_eye_gaze_features)
    dataloader = DataLoader(tokenized_dataset, shuffle=False, collate_fn=data_collator, batch_size=batch_size)

    predict_silver_labels(model, dataloader, output_path)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model_path')
    parser.add_argument('-u', '--user_id', dest='user_id', type=int)
    parser.add_argument('-t', '--downstream_task', dest='downstream_task', type=str, choices=['sentiment', 'complexity', 'cola', 'mnli', 'mrpc', 'qnli', 'qqp', 'rte', 'sst2', 'stsb', 'wnli'])
    args = parser.parse_args()

    output_dir = f'data/silver_labels/{args.downstream_task}'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if args.downstream_task == 'complexity':
        train_path = 'data/complexity/complexity_ds_en_train.csv'
        test_path = 'data/complexity/complexity_ds_en_test.csv'
        train_dataset = load_complexity_dataset(train_path)
        test_dataset = load_complexity_dataset(test_path)        
    elif args.downstream_task in list(TASK_TO_KEYS.keys()):
        dataset = load_dataset('nyu-mll/glue', args.downstream_task)
        train_dataset = dataset['train']
        if args.downstream_task == 'mnli':
            test_dataset ={
                'validation_matched': dataset['validation_matched'],
                'validation_mismatched': dataset['validation_mismatched']            
            }        
        else:
            test_dataset = dataset['validation']


    
    model = RobertaForMultiTaskTokenClassification.from_pretrained(args.model_path)
    tokenizer = AutoTokenizer.from_pretrained('FacebookAI/roberta-base', add_prefix_space=True)
    data_collator = DataCollatorForMultiTaskTokenClassification(tokenizer=tokenizer)
    
    train_output_path = os.path.join(output_dir, f'train_{args.user_id}.pkl')
    prepare_dataset_and_predict(train_dataset, args.downstream_task, tokenizer, train_output_path, model, data_collator)

    if args.downstream_task == 'mnli':
        for test_type in test_dataset.keys():
            test_output_path = os.path.join(output_dir,f'test_{test_type.split["_"][-1]}_{args.user_id}.pkl')
            prepare_dataset_and_predict(test_dataset[test_type], args.downstream_task, tokenizer, test_output_path, model, data_collator)
    else:
        test_output_path = os.path.join(output_dir, f'test_{args.user_id}.pkl')
        prepare_dataset_and_predict(test_dataset, args.downstream_task, tokenizer, test_output_path, model, data_collator)



if __name__ == '__main__':
    main()
