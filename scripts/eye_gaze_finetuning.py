import sys
import os
sys.path.append(os.path.abspath('.'))

from modules.data.dataset_utils import create_senteces_from_data, scale_datasets, tokenize_and_align_labels
from modules.modeling.custom_modeling_roberta import RobertaForMultiTaskTokenClassification
from modules.modeling.custom_data_collator import DataCollatorForMultiTaskTokenClassification
from transformers import AutoTokenizer, TrainingArguments, Trainer, set_seed, AutoConfig
import pandas as pd
import argparse
import evaluate


SEED = 42
TASKS = ['firstfix_dur','dur','firstrun_nfix','nfix','firstrun_dur']
 

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model_name', dest='model_name', type=str, default='FacebookAI/roberta-base')
    parser.add_argument('-u', '--user_id', dest='user_id', type=int, default=21)
    parser.add_argument('-b', '--batch_size', type=int, default=32)
    args = parser.parse_args()

    set_seed(42)

    train_path = f'data/geco/dataset/pp{args.user_id}_dataset_train.csv'
    test_path = f'data/geco/dataset/pp{args.user_id}_dataset_test.csv'

    train_df = pd.read_csv(train_path, index_col=0)
    test_df = pd.read_csv(test_path, index_col=0)
    
    train_dataset = create_senteces_from_data(train_df, TASKS)
    test_dataset = create_senteces_from_data(test_df, TASKS)

    # scale dataset features in [0, 100]
    train_dataset, test_dataset = scale_datasets(train_dataset, test_dataset)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, add_prefix_space=True)

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
    
    data_collator = DataCollatorForMultiTaskTokenClassification(tokenizer)
   
    metric = evaluate.load('rmse')
    def compute_metrics(eval_pred):
        res = dict()

        for task_idx, task in enumerate(trainer.label_names):
            labels = eval_pred.label_ids[task_idx].flatten()
            predictions = eval_pred.predictions[task[len('label_'):]].flatten()
            res[task] = metric.compute(predictions=predictions, references=labels)
        return res

    
    config = AutoConfig.from_pretrained(args.model_name)
    config.update({'tasks': TASKS, 'keys_to_ignore_at_inference':['mse_loss', 'mae_loss', 'labels']})
    model = RobertaForMultiTaskTokenClassification.from_pretrained(args.model_name, config=config)
    
    training_args = TrainingArguments(
        output_dir='models/prova', 
        eval_strategy='epoch',
        label_names=[f'label_{task}' for task in TASKS],
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8
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


if __name__ == '__main__':
    main()