import sys
import os
sys.path.append(os.path.abspath('.'))
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['WANDB_DISABLED'] = 'true'


from modules.data.dataset_utils import create_senteces_from_data, scale_datasets, tokenize_and_align_labels
from modules.modeling.custom_modeling_roberta import  RobertaForInterleavedMultitask, RobertaForSilverLabelMultitask
from transformers import AutoTokenizer, TrainingArguments, set_seed, AutoConfig
from modules.modeling.custom_trainer import InterleavedMultitaskFinetuningTrainer
from modules.modeling.custom_data_collator import DataCollatorForInterleavedMultiTask
from datasets import Dataset, load_dataset, load_from_disk
from itertools import cycle
import numpy as np
import pandas as pd
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


def load_complexity_dataset(src_path:str) -> Dataset:
    df = pd.read_csv(src_path)
    annotators_columns = [col for col in df.columns if col.startswith('judgement')]
    df['label'] = df[annotators_columns].mean(axis=1)
    df = df[['SENTENCE', 'label']]
    df = df.rename(columns={'label': 'label_complexity'})
    return Dataset.from_pandas(df)


def load_dst_dataset_complexity(tokenizer):
    train_path = 'data/complexity/complexity_ds_en_train.csv'
    test_path = 'data/complexity/complexity_ds_en_test.csv'

    
    train_dataset = load_complexity_dataset(train_path)
    test_dataset = load_complexity_dataset(test_path)


    def preprocess_function(examples):
        return tokenizer(examples['SENTENCE'], truncation=True)


    tokenized_train_dataset = train_dataset.map(preprocess_function, remove_columns=['SENTENCE'], desc="Running tokenizer on train dataset")
    tokenized_test_dataset = test_dataset.map(preprocess_function, remove_columns=['SENTENCE'], desc="Running tokenizer on dataset")

    return tokenized_train_dataset, tokenized_test_dataset


def load_dst_dataset_sentiment(tokenizer):
    dataset = load_dataset("sst2")
    dataset = dataset.rename_column("label", "label_sentiment")

    def preprocess_function(examples):
        return tokenizer(examples['sentence'], truncation=True, padding=True)

    tokenized_dataset = dataset.map(preprocess_function, remove_columns=['sentence', 'idx'], batched=True, desc="Running tokenizer on dataset")
    
    return tokenized_dataset['train'], tokenized_dataset['validation']

def load_dst_dataset_glue(task, tokenizer):
    # dataset = load_dataset('nyu-mll/glue', task)
    dataset = load_from_disk(os.path.join('data/glue', task))
    dataset = dataset.rename_column('label', f'label_{task}')

    def preprocess_dataset(downstream_task, dataset, tokenizer):
        sentence1_key, sentence2_key = TASK_TO_KEYS[downstream_task]

        def preprocess_function(examples):
            args = ((examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key]))
            return tokenizer(*args, padding=True, truncation=True)
        
        cols_to_remove = list(TASK_TO_KEYS[task]) if TASK_TO_KEYS[task][1] is not None else [TASK_TO_KEYS[task][0]]
        cols_to_remove.append('idx') 
        tokenized_dataset = dataset.map(preprocess_function, remove_columns=cols_to_remove, batched=True, desc="Running tokenizer on dataset")
        return tokenized_dataset
    
    tokenized_dataset = preprocess_dataset(task, dataset, tokenizer)
    train_dataset = tokenized_dataset['train']#.select(range(100))
    if task == 'mnli':
        eval_dataset ={
            'validation_matched': tokenized_dataset['validation_matched'],
            'validation_mismatched': tokenized_dataset['validation_mismatched']            
        }
    else:
        eval_dataset = tokenized_dataset['validation']
    return train_dataset, eval_dataset

def load_dst_dataset_coherence(tokenizer, text_domain):
    data_files = {'train': f'data/coherence/{text_domain}/en_train.tsv',
                  'validation': f'data/coherence/{text_domain}/en_eval.tsv'}
    
    dataset = load_dataset('csv', data_files=data_files, sep='\t')
    
    label_list = list(set(dataset['train']['label']))
    label_list.sort()

    label_to_id = {v: i for i, v in enumerate(label_list)}

    def preprocessing_function(examples):
        result = tokenizer(examples['text'], padding=True, truncation=True)
        result["label"] = [(label_to_id[l] if l != -1 else -1) for l in examples["label"]]
        return result

    tokenized_dataset = dataset.map(
        preprocessing_function,
        batched=True,
        remove_columns=['passage_id', 'text'],
        desc="Running tokenizer on dataset",
    )

    tokenized_dataset = tokenized_dataset.rename_column('label', f'label_coherence')

    return tokenized_dataset['train'], tokenized_dataset['validation']


def load_eye_gaze_datasets(user_id, tokenizer):
    train_path = f'data/geco/dataset/pp{user_id}_dataset_train.csv'
    test_path = f'data/geco/dataset/pp{user_id}_dataset_test.csv'

    train_df = pd.read_csv(train_path, index_col=0)
    test_df = pd.read_csv(test_path, index_col=0)
    
    train_dataset = create_senteces_from_data(train_df, TASKS)
    test_dataset = create_senteces_from_data(test_df, TASKS)

    # scale dataset features in [0, 100]
    train_dataset, test_dataset = scale_datasets(train_dataset, test_dataset)    

    tokenized_train_dataset = train_dataset.map(
                    tokenize_and_align_labels(tokenizer,[f'label_{task}' for task in TASKS]),
                    batched=True,
                    remove_columns=['text'],
                    desc="Running tokenizer on train dataset",
                    )
        
    tokenized_test_dataset = test_dataset.map(
                        tokenize_and_align_labels(tokenizer, [f'label_{task}' for task in TASKS]),
                        batched=True,
                        remove_columns=['text'],
                        desc="Running tokenizer on dataset",
                        )
    
    return tokenized_train_dataset, tokenized_test_dataset

def add_chunk(joint_data, chunk, batch_size):
    for label in joint_data:
        joint_data[label].extend(chunk.get(label, [None] * batch_size))


def get_chunk(iterator, keys, batch_size):
    chunk = {key: [] for key in keys}
    for _ in range(batch_size):
        next_item = next(iterator)
        for key in keys:
            chunk[key].append(next_item[key])
    return chunk

def join_datasets(eye_gaze_dataset, dst_dataset, batch_size):
    all_features = list(set(eye_gaze_dataset.column_names) | set(dst_dataset.column_names))
    iter_eye_gaze = cycle(eye_gaze_dataset)
    joint_data = {feat: [] for feat in all_features}

    # Iterate through dst_train in chunks of batch_size
    for i in range(0, len(dst_dataset), batch_size):
        if i + batch_size > len(dst_dataset):  # dobbiamo scartare il batch non pieno perchè il dataloader altrimenti carica dall'altro dataset
            break
        chunk_dst = dst_dataset[i:i+batch_size]
        add_chunk(joint_data, chunk_dst, batch_size)
        
        # Get corresponding chunk from eye_gaze_train
        batch_len = min(batch_size, len(chunk_dst['input_ids']))
        chunk_eye_gaze = get_chunk(iter_eye_gaze, eye_gaze_dataset.column_names, batch_len)
        add_chunk(joint_data, chunk_eye_gaze, batch_size)

    return Dataset.from_dict(joint_data)




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model_name', dest='model_name', type=str, default='FacebookAI/roberta-base')
    parser.add_argument('-u', '--user_id', dest='user_id', type=int, default=21)
    parser.add_argument('-b', '--batch_size', type=int, default=16)
    parser.add_argument('-l', '--learning_rate', dest='learning_rate', type=float, default=1e-05)
    parser.add_argument('-e', '--epochs', dest='training_epochs', type=int, default=10)
    parser.add_argument('-d', '--weight_decay', dest='weight_decay', type=float, default=0.1)
    parser.add_argument('-t', '--downstream_task', dest='downstream_task', type=str, choices=['sentiment', 'complexity', 'cola', 'mnli', 'mrpc', 'qnli', 'qqp', 'rte', 'sst2', 'stsb', 'wnli', 'coherence'])
    parser.add_argument('-x', '--text_domain', default='wiki', choices=['wiki', 'fanfic', 'ted', 'news'])
    parser.add_argument('-w', '--weighted_loss', dest='weighted_loss', action='store_true')
    parser.add_argument('-o', '--output_dir')
    args = parser.parse_args()

    set_seed(SEED)
    
    model_out_dir = args.output_dir

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, add_prefix_space=True)
    
    eye_gaze_train, eye_gaze_test = load_eye_gaze_datasets(args.user_id, tokenizer)    
    dst_label = f'label_{args.downstream_task}'

    if args.downstream_task == 'complexity':
        dst_train, dst_test = load_dst_dataset_complexity(tokenizer)
        downstream_type = 'regression'
        num_labels = 1
    elif args.downstream_task == 'sentiment':
        dst_train, dst_test = load_dst_dataset_sentiment(tokenizer)
        downstream_type = 'classification'
        num_labels = 2
    elif args.downstream_task == 'coherence':
        dst_train, dst_test = load_dst_dataset_coherence(tokenizer, args.text_domain)
        downstream_type = 'classification'
        num_labels = 11
    elif args.downstream_task in list(TASK_TO_KEYS.keys()):
        dst_train, dst_test = load_dst_dataset_glue(args.downstream_task, tokenizer)
        if args.downstream_task == 'stsb':
            downstream_type = 'regression'
            num_labels = 1
        else:
            num_labels = len(set(dst_train[dst_label]))
            downstream_type = 'classification'
    else:
        raise Exception(f'Downstream task {args.downstream_task} not supported.')


    train_dataset = join_datasets(eye_gaze_train, dst_train, args.batch_size)
    if args.downstream_task == 'mnli':
        test_dataset = {'eye_gaze': eye_gaze_test, 'dst_matched': dst_test['validation_matched'], 'dst_mismatched': dst_test['validation_mismatched']}
    else:
        test_dataset = {'eye_gaze': eye_gaze_test, 'dst': dst_test}

    data_collator = DataCollatorForInterleavedMultiTask(tokenizer, dst_label, [f'label_{task}' for task in TASKS])
    
    mae = evaluate.load('mae')
    spearmanr = evaluate.load("spearmanr")
    accuracy = evaluate.load("accuracy")
    if args.downstream_task in list(TASK_TO_KEYS.keys()):
        glue_metric = evaluate.load("glue", args.downstream_task)

    def compute_metrics_eye_gaze(eval_pred):
        res = dict()
        for task_idx, task in enumerate([f'label_{task}' for task in TASKS]):
            labels = eval_pred.label_ids[task_idx].flatten()
            predictions = eval_pred.predictions[task[len('label_'):]].squeeze().flatten()
            
            not_masked_labels = labels != -100
            labels = labels[not_masked_labels]
            predictions = predictions[not_masked_labels]

            res[task] = {
                'mae': mae.compute(predictions=predictions, references=labels)['mae'],
                'spearmanr': spearmanr.compute(predictions=predictions, references=labels)['spearmanr']
            }
        return res

    def compute_metrics_complexity(eval_pred):
        logits, labels = eval_pred
        logits = logits['complexity']
        labels = labels.reshape(-1, 1)

        res = {
                'mae': mae.compute(predictions=logits, references=labels)['mae'],
                'spearmanr': spearmanr.compute(predictions=logits, references=labels)['spearmanr']
            }           
        return res
    
    def compute_metrics_coherence(eval_pred):
        preds = eval_pred.predictions['coherence'][0] if isinstance(eval_pred.predictions['coherence'], tuple) else eval_pred.predictions['coherence']
        preds = np.argmax(preds, axis=1)
        result = accuracy.compute(predictions=preds, references=eval_pred.label_ids)
        if len(result) > 1:
            result["combined_score"] = np.mean(list(result.values())).item()

        return result

    def compute_metrics_accuracy(eval_pred):
        logits, labels = eval_pred
        logits = logits['sentiment']

        predictions = np.argmax(logits, axis=-1)
        return accuracy.compute(predictions=predictions, references=labels)
    
    def compute_metrics_glue(p):
        predictions = p.predictions[args.downstream_task]
        preds = predictions[0] if isinstance(predictions, tuple) else predictions
        preds = np.squeeze(preds) if downstream_type == 'regression' else np.argmax(preds, axis=1)
        result = glue_metric.compute(predictions=preds, references=p.label_ids)
        if len(result) > 1:
            result["combined_score"] = np.mean(list(result.values())).item()
        return result
    
    if args.downstream_task == 'complexity':
        compute_metrics = {'eye_gaze': compute_metrics_eye_gaze, 'dst': compute_metrics_complexity}
    elif args.downstream_task == 'sentiment':
        compute_metrics = {'eye_gaze': compute_metrics_eye_gaze, 'dst': compute_metrics_accuracy}
    elif args.downstream_task == 'coherence':
        compute_metrics = {'eye_gaze': compute_metrics_eye_gaze, 'dst': compute_metrics_coherence}
    elif args.downstream_task == 'mnli':
        compute_metrics = {'eye_gaze': compute_metrics_eye_gaze, 'dst_matched': compute_metrics_glue, 'dst_mismatched': compute_metrics_glue}
    elif args.downstream_task in list(TASK_TO_KEYS.keys()):
        compute_metrics = {'eye_gaze': compute_metrics_eye_gaze, 'dst': compute_metrics_glue}
    else:
        raise Exception(f'Downstream task {args.downstream_task} not supported.')

    
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
        save_strategy='no',
        warmup_ratio=0.06
        )
    

    trainer = InterleavedMultitaskFinetuningTrainer(
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