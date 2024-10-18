import sys
import os
sys.path.append(os.path.abspath('.'))
os.environ["CUDA_VISIBLE_DEVICES"] = "0" 

from modules.modeling.model_utils import get_tokenizer_name
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from transformers import (
    AutoModelForSequenceClassification, 
    DataCollatorWithPadding,
    TrainingArguments, 
    AutoTokenizer, 
    set_seed, 
    Trainer,    
)

from datasets import Dataset, load_dataset
import pandas as pd
import numpy as np
import argparse
import evaluate


SEED = 42 


def load_complexity_dataset(src_path:str) -> Dataset:
    df = pd.read_csv(src_path)
    annotators_columns = [col for col in df.columns if col.startswith('judgement')]
    df['label'] = df[annotators_columns].mean(axis=1)
    df = df[['SENTENCE', 'label']]
    # df = df.rename(columns={'SENTENCE': 'text'})
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

    def preprocess_function(examples):
        return tokenizer(examples['sentence'], truncation=True, padding=True)

    tokenized_dataset = dataset.map(preprocess_function, remove_columns=['sentence', 'idx'], batched=True, desc="Running tokenizer on dataset")
    
    return tokenized_dataset['train'], tokenized_dataset['validation']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model_path', dest='model_path')
    parser.add_argument('-b', '--batch_size', type=int, default=16)
    parser.add_argument('-t', '--downstream_task', dest='downstream_task', type=str, choices=['complexity', 'sentiment'])
    parser.add_argument('-l', '--learning_rate', dest='learning_rate', type=float, default=5e-05)
    parser.add_argument('-e', '--epochs', dest='training_epochs', type=int, default=10)
    parser.add_argument('-d', '--weight_decay', dest='weight_decay', type=float, default=1.0)
    parser.add_argument('-o', '--output_dir')
    args = parser.parse_args()

    set_seed(SEED)

    model_string = '_'.join(args.model_path.split('/')[-1].split('_')[:2])
    tokenizer_name = get_tokenizer_name(model_string)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, add_prefix_space=True)

    output_dir = args.output_dir #f'models/{args.downstream_task}/lora/{model_string}'

    if args.downstream_task == 'complexity':
        train_dataset, test_dataset = load_dst_dataset_complexity(tokenizer)
        num_labels = 1
        mae = evaluate.load('mae')
        spearmanr = evaluate.load("spearmanr")
        def compute_metrics(eval_pred):
            logits, labels = eval_pred
            labels = labels.reshape(-1, 1)
            res = {
                    'mae': mae.compute(predictions=logits, references=labels)['mae'],
                    'spearmanr': spearmanr.compute(predictions=logits, references=labels)['spearmanr']
                }            
            return res
    elif args.downstream_task == 'sentiment':
        train_dataset, test_dataset = load_dst_dataset_sentiment(tokenizer)
        num_labels = 2
        accuracy = evaluate.load("accuracy")
        def compute_metrics(eval_pred):
            logits, labels = eval_pred
            predictions = np.argmax(logits, axis=-1)
            return accuracy.compute(predictions=predictions, references=labels)
    else:
        raise Exception(f'Task {args.downstream_task} not implemented.')
    
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    
    model = AutoModelForSequenceClassification.from_pretrained(args.model_path, num_labels=num_labels)
    

    lora_config = LoraConfig(
        r=32,
        lora_alpha=8,
        target_modules=["query", "value"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.SEQ_CLS, # this is necessary
        inference_mode=False
    )

    model = get_peft_model(model, lora_config)

    training_args = TrainingArguments(
        output_dir=output_dir+'_adapters', 
        eval_strategy='epoch',
        logging_strategy='epoch',
        logging_dir=output_dir+'_adapters',
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.training_epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_steps=500,
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
    trainer.save_model(output_dir+'_adapters')
    trainer.save_state()

    original_model = AutoModelForSequenceClassification.from_pretrained(args.model_path, num_labels=num_labels)
    original_with_adapter = PeftModel.from_pretrained(original_model, output_dir+'_adapters')
    merged_model = original_with_adapter.merge_and_unload()
    merged_model.save_pretrained(output_dir)    
    


if __name__ == '__main__':
    main()