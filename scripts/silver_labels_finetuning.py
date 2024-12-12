import sys
import os
sys.path.append(os.path.abspath('.'))
os.environ["CUDA_VISIBLE_DEVICES"] = "0" 

from transformers import AutoTokenizer, TrainingArguments, set_seed, AutoConfig, Trainer
from modules.modeling.custom_modeling_roberta import  RobertaForSilverLabelMultitask
from modules.modeling.custom_data_collator import DataCollatorForMultiTask
from datasets import Dataset, load_dataset, load_from_disk
import pandas as pd
import numpy as np
import argparse
import evaluate


SEED = 42
TASKS = ['firstfix_dur','dur','firstrun_nfix','nfix','firstrun_dur']


TASK_TO_KEYS = {
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
 
def load_complexity_dataframe(src_path:str) -> pd.DataFrame:
    df = pd.read_csv(src_path)
    annotators_columns = [col for col in df.columns if col.startswith('judgement')]
    df['label'] = df[annotators_columns].mean(axis=1)
    df = df[['SENTENCE', 'label']]
    df = df.rename(columns={'label': 'label_complexity'})
    return df


def join_dataframes(dst_df, silver_eye_gaze_labels):
    return pd.concat([dst_df, silver_eye_gaze_labels], axis=1)


def load_silver_labels_df(src_path: str) -> pd.DataFrame:
    df = pd.read_pickle(src_path)
    df = df.rename(columns=lambda col: 'label_' + col)
    return df


def prepare_complexity_datasets(train_silver_labels_df, test_silver_labels_df, tokenizer):
    train_path = 'data/complexity/complexity_ds_en_train.csv'
    test_path = 'data/complexity/complexity_ds_en_test.csv'

    dst_train_df = load_complexity_dataframe(train_path)
    dst_test_df = load_complexity_dataframe(test_path)

    train_df_joined = join_dataframes(dst_train_df, train_silver_labels_df)
    test_df_joined = join_dataframes(dst_test_df, test_silver_labels_df)

    train_dataset = Dataset.from_pandas(train_df_joined)
    test_dataset = Dataset.from_pandas(test_df_joined)

    def preprocess_function(examples):
        return tokenizer(examples['SENTENCE'], truncation=True)

    tokenized_train_dataset = train_dataset.map(preprocess_function, remove_columns=['SENTENCE'], desc="Running tokenizer on train dataset")
    tokenized_test_dataset = test_dataset.map(preprocess_function, remove_columns=['SENTENCE'], desc="Running tokenizer on train dataset")

    return tokenized_train_dataset, tokenized_test_dataset

def prepare_sentiment_datasets(train_silver_labels_df, test_silver_labels_df, tokenizer):
    datasets =  load_dataset("sst2")
    datasets = datasets.rename_column("label", "label_sentiment")

    dst_train_df = datasets['train'].to_pandas()
    dst_test_df = datasets['validation'].to_pandas()

    train_df_joined = join_dataframes(dst_train_df, train_silver_labels_df)
    test_df_joined = join_dataframes(dst_test_df, test_silver_labels_df)

    train_dataset = Dataset.from_pandas(train_df_joined)
    test_dataset = Dataset.from_pandas(test_df_joined)


    def preprocess_function(examples):
        return tokenizer(examples['sentence'], truncation=True, padding=True)

    tokenized_train_dataset = train_dataset.map(preprocess_function, remove_columns=['sentence', 'idx'], desc="Running tokenizer on train dataset")
    tokenized_test_dataset = test_dataset.map(preprocess_function, remove_columns=['sentence', 'idx'], desc="Running tokenizer on train dataset")

    return tokenized_train_dataset, tokenized_test_dataset


def prepare_glue_dataset(task, train_silver_labels_df, test_silver_labels_df, tokenizer):
    # dataset = load_dataset('nyu-mll/glue', task)
    datasets = load_from_disk(os.path.join('data/glue', task))
    datasets = datasets.rename_column('label', f'label_{task}')

    dst_train_df = datasets['train'].to_pandas()
    train_df_joined = join_dataframes(dst_train_df, train_silver_labels_df)
    train_dataset = Dataset.from_pandas(train_df_joined)

    if task == 'mnli':
        dst_test_df ={
            'validation_matched': datasets['validation_matched'].to_pandas(),
            'validation_mismatched': datasets['validation_mismatched'].to_pandas()   
        }
        test_df_joined = {test_type: join_dataframes(test_df, test_silver_labels_df[test_type]) for test_type, test_df in dst_test_df.items()}
        test_dataset = {test_type: Dataset.from_pandas(test_df) for test_type, test_df in test_df_joined.items()}
    else:
        dst_test_df = datasets['validation'].to_pandas()
        test_df_joined = join_dataframes(dst_test_df, test_silver_labels_df)
        test_dataset = Dataset.from_pandas(test_df_joined)



    def preprocess_dataset(downstream_task, dataset, tokenizer):
        sentence1_key, sentence2_key = TASK_TO_KEYS[downstream_task]

        def preprocess_function(examples):
            args = ((examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key]))
            return tokenizer(*args, padding=True, truncation=True)
        
        cols_to_remove = list(TASK_TO_KEYS[task]) if TASK_TO_KEYS[task][1] is not None else [TASK_TO_KEYS[task][0]]
        cols_to_remove.append('idx') 
        tokenized_dataset = dataset.map(preprocess_function, remove_columns=cols_to_remove, batched=True, desc="Running tokenizer on dataset")
        return tokenized_dataset
    
    tokenized_train_dataset = preprocess_dataset(task, train_dataset, tokenizer)
    if task == 'mnli':
        tokenized_test_dataset = {test_type: preprocess_dataset(task, test_dataset, tokenizer) for test_type, test_dataset in test_dataset.items()}
    else:
        tokenized_test_dataset = preprocess_dataset(task, test_dataset, tokenizer)
   
    return tokenized_train_dataset, tokenized_test_dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model_name', dest='model_name', type=str, default='FacebookAI/roberta-base')
    parser.add_argument('-u', '--user_id', dest='user_id', type=int, default=21)
    parser.add_argument('-b', '--batch_size', type=int, default=16)
    parser.add_argument('-l', '--learning_rate', dest='learning_rate', type=float, default=1e-05)
    parser.add_argument('-e', '--epochs', dest='training_epochs', type=int, default=10)
    parser.add_argument('-d', '--weight_decay', dest='weight_decay', type=float, default=1.0)
    parser.add_argument('-t', '--downstream_task', dest='downstream_task', type=str, choices=['sentiment', 'complexity', 'cola', 'mnli', 'mrpc', 'qnli', 'qqp', 'rte', 'sst2', 'stsb', 'wnli'])
    parser.add_argument('-o', '--output_path')
    args = parser.parse_args()

    set_seed(SEED)


    model_out_dir = args.output_path #f'models/silver_labels/{args.downstream_task}/{model_string}_pp{args.user_id}'
    train_silver_labels_path = f'data/silver_labels/{args.downstream_task}/train_{args.user_id}.pkl'

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, add_prefix_space=True)
    
    train_silver_labels_df = load_silver_labels_df(train_silver_labels_path)
    if args.downstream_task == 'mnli':
        test_silver_labels_matched_path = f'data/silver_labels/{args.downstream_task}/test_matched_{args.user_id}.pkl'
        test_silver_labels_mismatched_path = f'data/silver_labels/{args.downstream_task}/test_mismatched_{args.user_id}.pkl'
        test_silver_labels_df ={
            'validation_matched': load_silver_labels_df(test_silver_labels_matched_path),
            'validation_mismatched': load_silver_labels_df(test_silver_labels_mismatched_path)
        }
    else:
        test_silver_labels_path = f'data/silver_labels/{args.downstream_task}/test_{args.user_id}.pkl'
        test_silver_labels_df = load_silver_labels_df(test_silver_labels_path)

    dst_label = f'label_{args.downstream_task}'

    if args.downstream_task == 'complexity':
        downstream_type = 'regression'
        num_labels = 1
        train_dataset, test_dataset = prepare_complexity_datasets(train_silver_labels_df, test_silver_labels_df, tokenizer)
    elif args.downstream_task == 'sentiment':
        downstream_type = 'classification'
        num_labels = 2
        train_dataset, test_dataset =  prepare_sentiment_datasets(train_silver_labels_df, test_silver_labels_df, tokenizer)
    elif args.downstream_task in list(TASK_TO_KEYS.keys()):
        train_dataset, test_dataset = prepare_glue_dataset(args.downstream_task, train_silver_labels_df, test_silver_labels_df, tokenizer)
        if args.downstream_task == 'stsb':
            downstream_type = 'regression'
            num_labels = 1
        else:
            num_labels = len(set(train_dataset[dst_label]))
            downstream_type = 'classification'


    data_collator = DataCollatorForMultiTask(dst_label=dst_label, eye_gaze_labels=[f'label_{task}' for task in TASKS], tokenizer=tokenizer)
    
   
    mae = evaluate.load('mae')
    spearmanr = evaluate.load('spearmanr')
    if args.downstream_task == 'sentiment':
        glue_metric = evaluate.load("glue", 'sst2')
    elif args.downstream_task in list(TASK_TO_KEYS.keys()):
        glue_metric = evaluate.load("glue", args.downstream_task)

    def compute_metrics(eval_pred):
        res = dict()
        for task_idx, task in enumerate([f'{task}' for task in TASKS] + [args.downstream_task]):
            if task not in list(TASK_TO_KEYS.keys())+['sentiment']:
                labels = eval_pred.label_ids[task_idx].flatten()
                predictions = eval_pred.predictions[task].squeeze().flatten()
                if task != 'complexity':
                    not_masked_labels = labels != -100
                    labels = labels[not_masked_labels]
                    predictions = predictions[not_masked_labels]
                res[task] = {
                    'mae': mae.compute(predictions=predictions, references=labels)['mae'],
                    'spearmanr': spearmanr.compute(predictions=predictions, references=labels)['spearmanr']
                }
            else:
                predictions = eval_pred.predictions[args.downstream_task]
                preds = predictions[0] if isinstance(predictions, tuple) else predictions
                preds = np.squeeze(preds) if downstream_type == 'regression' else np.argmax(preds, axis=1)
                result = glue_metric.compute(predictions=preds, references=eval_pred.label_ids[task_idx].flatten())
                if len(result) > 1:
                    result["combined_score"] = np.mean(list(result.values())).item()
                res[task] = {metric: score for metric, score in result.items()}

        return res
    
    
    config = AutoConfig.from_pretrained(args.model_name)
    config.update({'token_tasks': TASKS,  'donwstream_task': args.downstream_task, 'downstream_type': downstream_type, 'keys_to_ignore_at_inference':['mse_loss', 'mae_loss', 'labels']})
    config.num_labels = num_labels

    model = RobertaForSilverLabelMultitask.from_pretrained(args.model_name, config=config)

    training_args = TrainingArguments(
        output_dir=model_out_dir, 
        eval_strategy='epoch',
        logging_strategy='epoch',
        logging_dir=model_out_dir,
        label_names=[f'label_{task}' for task in TASKS+[args.downstream_task]],
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.training_epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        save_strategy='no'
        )
    

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )


    trainer.train()
    trainer.save_model(model_out_dir)
    trainer.save_state()


if __name__ == '__main__':
    main()