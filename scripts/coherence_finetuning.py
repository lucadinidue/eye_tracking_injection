import sys
import os
sys.path.append(os.path.abspath('.'))
os.environ["CUDA_VISIBLE_DEVICES"] = "0" 
os.environ['WANDB_DISABLED'] = 'true'

from modules.modeling.model_utils import get_tokenizer_name
from transformers import (
    AutoModelForSequenceClassification, 
    DataCollatorWithPadding,
    TrainingArguments, 
    AutoTokenizer, 
    set_seed, 
    Trainer,    
)

from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from datasets import Dataset, load_dataset, load_from_disk
import pandas as pd
import numpy as np
import argparse
import evaluate


FREEZE_LAYERS_MAP = {
    'regressor_only': ['classifier'],
    'last_2': ['classifier', 'encoder.layer.11', ],
    'last_3': ['classifier', 'encoder.layer.11', 'encoder.layer.10']

}


SEED = 42 



def preprocess_dataset(dataset, label_to_id, tokenizer):

    def preprocessing_function(examples):
        result = tokenizer(examples['text'], padding='max_length', max_length=256, truncation=True)
        result["label"] = [(label_to_id[l] if l != -1 else -1) for l in examples["label"]]
        return result

    tokenized_dataset = dataset.map(
        preprocessing_function,
        batched=True,
        remove_columns=['passage_id', 'text'],
        desc="Running tokenizer on dataset",
    )
    return tokenized_dataset

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model_path', dest='model_path')
    parser.add_argument('-b', '--batch_size', type=int, default=32)
    parser.add_argument('-l', '--learning_rate', dest='learning_rate', type=float, default=5e-05)
    parser.add_argument('-e', '--epochs', dest='training_epochs', type=int)
    parser.add_argument('-d', '--weight_decay', dest='weight_decay', type=float, default=0.1)
    parser.add_argument('-f', '--freeze_layers', type=str, default=None)
    parser.add_argument('-o', '--output_dir')
    parser.add_argument('-t', '--type', type=str, choices=['ted', 'wiki', 'news', 'fanfic'])
    args = parser.parse_args()

    set_seed(SEED)

    output_dir = args.output_dir 

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    data_files = {'train': f'data/coherence/{args.type}/en_train.tsv',
                  'validation': f'data/coherence/{args.type}/en_eval.tsv'}
    
    dataset = load_dataset('csv', data_files=data_files, sep='\t')
    
    label_list = list(set(dataset['train']['label']))
    label_list.sort()
    num_labels = len(label_list)

    tokenizer_name = get_tokenizer_name('roberta')
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, add_prefix_space=True)   
    model = AutoModelForSequenceClassification.from_pretrained(args.model_path, num_labels=num_labels)

    label_to_id = {v: i for i, v in enumerate(label_list)}
    model.config.label2id = label_to_id
    model.config.id2label = {label_id: label for label, label_id in model.config.label2id.items()}

    train_dataset = preprocess_dataset(dataset['train'], label_to_id, tokenizer, )
    eval_dataset = preprocess_dataset(dataset['validation'], label_to_id, tokenizer)

    
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    metric = evaluate.load('accuracy')  # , cache_dir=training_args.cache_dir)

    def compute_metrics(p):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        preds = np.argmax(preds, axis=1)
        result = metric.compute(predictions=preds, references=p.label_ids)
        if len(result) > 1:
            result["combined_score"] = np.mean(list(result.values())).item()

        return result

    if args.freeze_layers == 'lora':
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
        output_dir = output_dir + '_adapters'

    elif args.freeze_layers is not None:
        not_freeze_weights = FREEZE_LAYERS_MAP[args.freeze_layers]
        for name, param in model.named_parameters():
            requires_grad = False
            for not_freeze_str in not_freeze_weights:
                if not_freeze_str in name:
                    requires_grad = True  
            param.requires_grad = requires_grad


    training_args = TrainingArguments(
        output_dir=output_dir, 
        eval_strategy='epoch',
        logging_strategy='epoch',
        logging_dir=output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.training_epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=0.06,
        save_strategy='no' 
        )


    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.save_model(output_dir)
    trainer.save_state()

    if args.freeze_layers == 'lora':
        original_model = AutoModelForSequenceClassification.from_pretrained(args.model_path, num_labels=num_labels)
        original_with_adapter = PeftModel.from_pretrained(original_model, output_dir)
        merged_model = original_with_adapter.merge_and_unload()
        merged_model.save_pretrained(output_dir[:-len('_adapters')])   


if __name__ == '__main__':
    main()
