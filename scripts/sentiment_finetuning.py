import sys
import os
sys.path.append(os.path.abspath('.'))
os.environ["CUDA_VISIBLE_DEVICES"] = "0" 

from modules.modeling.model_utils import get_tokenizer_name
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


FREEZE_LAYERS_MAP = {
    'regressor_only': ['classifier'],
    'last_2': ['classifier', 'encoder.layer.11', ],
    'last_3': ['classifier', 'encoder.layer.11', 'encoder.layer.10']

}

SEED = 42 


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model_path', dest='model_path')
    parser.add_argument('-b', '--batch_size', type=int, default=32)
    parser.add_argument('-l', '--learning_rate', dest='learning_rate', type=float, default=5e-05)
    parser.add_argument('-e', '--epochs', dest='training_epochs', type=int, default=3)
    parser.add_argument('-d', '--weight_decay', dest='weight_decay', type=float, default=1.0)
    parser.add_argument('-f', '--freeze_layers', type=str, default=None)
    parser.add_argument('-o', '--output_dir')
    args = parser.parse_args()

    set_seed(SEED)

    model_string = '_'.join(args.model_path.split('/')[-1].split('_')[:2])
    output_dir = args.output_dir#os.path.join('models/sentiment/prova_freeze', model_string)

    dataset = load_dataset("sst2")
    

    tokenizer_name = get_tokenizer_name(model_string)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, add_prefix_space=True)

    def preprocess_function(examples):
        return tokenizer(examples['sentence'], truncation=True, padding=True)


    tokenized_dataset = dataset.map(preprocess_function, batched=True, desc="Running tokenizer on dataset")
    
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    


    accuracy = evaluate.load("accuracy")
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return accuracy.compute(predictions=predictions, references=labels)

    
    model = AutoModelForSequenceClassification.from_pretrained(args.model_path, num_labels=2)


    if args.freeze_layers is not None:
        # output_dir += '_' + args.freeze_layers
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
        warmup_steps=500,
        save_strategy='no'
        )
    

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset['train'],
        eval_dataset=tokenized_dataset['validation'],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.save_model(output_dir)
    trainer.save_state()


if __name__ == '__main__':
    main()