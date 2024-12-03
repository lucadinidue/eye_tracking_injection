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


SEED = 42 



def preprocess_dataset(downstream_task, dataset, label_to_id, tokenizer):
    sentence1_key, sentence2_key = TASK_TO_KEYS[downstream_task]

    def preprocess_function(examples):
        args = (
            (examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
        )
        result = tokenizer(*args, padding=True, truncation=True)
        
        return result

    tokenized_dataset = dataset.map(preprocess_function, batched=True, desc="Running tokenizer on dataset")
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
    parser.add_argument('-t', '--downstream_task', type=str, choices=['cola', 'mnli', 'mrpc', 'qnli', 'qqp', 'rte', 'sst2', 'stsb', 'wnli'])
    args = parser.parse_args()

    set_seed(SEED)

    model_string = '_'.join(args.model_path.split('/')[-1].split('_')[:2])
    output_dir = args.output_dir #os.path.join('models/sentiment/prova_freeze', model_string)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    
    dataset = load_dataset('nyu-mll/glue', args.downstream_task)
    # dataset.save_to_disk(os.path.join('data/glue', args.downstream_task))
    # dataset = load_from_disk(os.path.join('data/glue', args.downstream_task))

    is_regression = args.downstream_task == "stsb"
    if not is_regression:
        label_list = dataset['train'].features['label'].names
        num_labels = len(label_list)
        label_to_id = {v: i for i, v in enumerate(label_list)}
    else:
        num_labels = 1
        label_to_id = None
        

    tokenizer_name = get_tokenizer_name(model_string)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, add_prefix_space=True)   


    tokenized_dataset = preprocess_dataset(args.downstream_task, dataset, label_to_id, tokenizer)

    
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    metric = evaluate.load("glue", args.downstream_task)

    def compute_metrics(p):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        preds = np.squeeze(preds) if is_regression else np.argmax(preds, axis=1)
        result = metric.compute(predictions=preds, references=p.label_ids)
        if len(result) > 1:
            result["combined_score"] = np.mean(list(result.values())).item()
        return result

    
    model = AutoModelForSequenceClassification.from_pretrained(args.model_path, num_labels=num_labels)

    if not is_regression:
        model.config.label2id = label_to_id
        model.config.id2label = {id: label for label, id in label_to_id.items()}


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

    
    train_dataset = tokenized_dataset['train']
    if args.downstream_task == 'mnli':
        eval_dataset ={
            'validation_matched': tokenized_dataset['validation_matched'],
            'validation_mismatched': tokenized_dataset['validation_mismatched']            
        }
    else:
        eval_dataset = tokenized_dataset['validation']

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
