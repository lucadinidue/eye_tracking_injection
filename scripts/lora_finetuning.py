import sys
import os
sys.path.append(os.path.abspath('.'))

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

from datasets import Dataset
import pandas as pd
import argparse
import evaluate


FREEZE_LAYERS_MAP = {
    'regressor_only': ['classifier'],
    'last_2': ['classifier', 'encoder.layer.11', ],
    'last_3': ['classifier', 'encoder.layer.11', 'encoder.layer.10']

}

SEED = 42 

def load_complexity_dataset(src_path:str) -> Dataset:
    df = pd.read_csv(src_path)
    annotators_columns = [col for col in df.columns if col.startswith('judgement')]
    df['label'] = df[annotators_columns].mean(axis=1)
    df = df[['SENTENCE', 'label']]
    df = df.rename(columns={'SENTENCE': 'text'})
    return Dataset.from_pandas(df)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model_path', dest='model_path')
    parser.add_argument('-b', '--batch_size', type=int, default=16)
    parser.add_argument('-l', '--learning_rate', dest='learning_rate', type=float, default=5e-05)
    parser.add_argument('-e', '--epochs', dest='training_epochs', type=int, default=10)
    parser.add_argument('-d', '--weight_decay', dest='weight_decay', type=float, default=1.0)
    parser.add_argument('-f', '--freeze_layers', type=str, default=None)
    args = parser.parse_args()

    set_seed(SEED)

    model_string = '_'.join(args.model_path.split('/')[-1].split('_')[:2])

    train_path = 'data/complexity/complexity_ds_en_train.csv'
    test_path = 'data/complexity/complexity_ds_en_test.csv'

    output_dir = os.path.join('models/lora_complexity', model_string)

    
    train_dataset = load_complexity_dataset(train_path)
    test_dataset = load_complexity_dataset(test_path)

    tokenizer_name = get_tokenizer_name(model_string)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, add_prefix_space=True)

    def preprocess_function(examples):
        return tokenizer(examples['text'], truncation=True)


    tokenized_train_dataset = train_dataset.map(preprocess_function, desc="Running tokenizer on train dataset")
    tokenized_test_dataset = test_dataset.map(preprocess_function, desc="Running tokenizer on dataset")
    
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    


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

    
    model = AutoModelForSequenceClassification.from_pretrained(args.model_path, num_labels=1)
    

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
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_test_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.save_model(output_dir+'_adapters')
    trainer.save_state()

    original_model = AutoModelForSequenceClassification.from_pretrained(args.model_path, num_labels=1)
    original_with_adapter = PeftModel.from_pretrained(original_model, output_dir+'_adapters')
    merged_model = original_with_adapter.merge_and_unload()
    merged_model.save_pretrained(output_dir)    
    


if __name__ == '__main__':
    main()