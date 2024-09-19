import sys
import os
sys.path.append(os.path.abspath('.'))

from transformers import AutoTokenizer, TrainingArguments, set_seed, AutoConfig, Trainer
from modules.modeling.custom_modeling_roberta import  RobertaForInterleavedMultitask
from modules.modeling.custom_data_collator import DataCollatorForMultiTask
from datasets import Dataset
import pandas as pd
import argparse
import evaluate


SEED = 42
TASKS = ['firstfix_dur','dur','firstrun_nfix','nfix','firstrun_dur']
 
def load_complexity_dataframe(src_path:str) -> pd.DataFrame:
    df = pd.read_csv(src_path)
    annotators_columns = [col for col in df.columns if col.startswith('judgement')]
    df['label'] = df[annotators_columns].mean(axis=1)
    df = df[['SENTENCE', 'label']]
    df = df.rename(columns={'label': 'label_complexity'})
    return df


def join_dataframes(dst_df, silver_eye_gaze_labels):
    return pd.concat([dst_df, silver_eye_gaze_labels], axis=1)


def join_dataframes(dst_df, silver_eye_gaze_labels):
    return pd.concat([dst_df, silver_eye_gaze_labels], axis=1)


def load_silver_labels_df(src_path: str) -> pd.DataFrame:
    df = pd.read_pickle(src_path)
    df = df.rename(columns=lambda col: 'label_' + col)
    return df


def prepare_complexity_datasets(train_path, test_path, train_silver_labels_df, test_silver_labels_df, tokenizer):
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model_name', dest='model_name', type=str, default='FacebookAI/roberta-base')
    parser.add_argument('-u', '--user_id', dest='user_id', type=int, default=21)
    parser.add_argument('-b', '--batch_size', type=int, default=16)
    parser.add_argument('-l', '--learning_rate', dest='learning_rate', type=float, default=1e-05)
    parser.add_argument('-e', '--epochs', dest='training_epochs', type=int, default=10)
    parser.add_argument('-d', '--weight_decay', dest='weight_decay', type=float, default=1.0)
    parser.add_argument('-t', '--downstream_task', dest='downstream_task', type=str, choices=['sentiment', 'complexity'])
    args = parser.parse_args()

    set_seed(SEED)

    model_string = args.model_name.split('/')[-1]
    

    model_out_dir = f'models/silver_labels/{args.downstream_task}/{model_string}_pp{args.user_id}'
    train_silver_labels_path = f'data/silver_labels/{args.downstream_task}/train_{args.user_id}.pkl'
    test_silver_labels_path = f'data/silver_labels/{args.downstream_task}/test_{args.user_id}.pkl'

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, add_prefix_space=True)
    
    train_silver_labels_df = load_silver_labels_df(train_silver_labels_path)
    test_silver_labels_df = load_silver_labels_df(test_silver_labels_path)


    if args.downstream_task == 'complexity':
        train_path = 'data/complexity/complexity_ds_en_train.csv'
        test_path = 'data/complexity/complexity_ds_en_test.csv'
        downstream_type = 'regression'
        dst_label = 'label_complexity'
        num_labels = 1
        train_dataset, test_dataset = prepare_complexity_datasets(train_path, test_path, train_silver_labels_df, test_silver_labels_df, tokenizer)

    else:
        raise Exception(f'Downstream task {args.downstream_task} not implemented yet!')
        dst_train, dst_test = None, None
        downstream_type = None


    data_collator = DataCollatorForMultiTask(dst_label=dst_label, eye_gaze_labels=[f'label_{task}' for task in TASKS], tokenizer=tokenizer)
    
   
    mae = evaluate.load('mae')
    spearmanr = evaluate.load("spearmanr")
    def compute_metrics(eval_pred):
        res = dict()
        for task_idx, task in enumerate([f'{task}' for task in TASKS] + ['complexity']):
            # print('task idx - task')
            # print(task_idx, task)
            labels = eval_pred.label_ids[task_idx].flatten()
            predictions = eval_pred.predictions[task].squeeze().flatten()
            # print('labels', len(labels))
            # print('predictions', len(predictions))
            if task_idx != 'complexity':
                not_masked_labels = labels != -100
                labels = labels[not_masked_labels]
                predictions = predictions[not_masked_labels]

            res[task] = {
                'mae': mae.compute(predictions=predictions, references=labels)['mae'],
                'spearmanr': spearmanr.compute(predictions=predictions, references=labels)['spearmanr']
            }
        return res
    
    
    config = AutoConfig.from_pretrained(args.model_name)
    config.update({'token_tasks': TASKS,  'donwstream_task': args.downstream_task, 'downstream_type': downstream_type, 'keys_to_ignore_at_inference':['mse_loss', 'mae_loss', 'labels']})
    config.num_labels = num_labels

    model = RobertaForInterleavedMultitask.from_pretrained(args.model_name, config=config)

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
        save_strategy='epoch',
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