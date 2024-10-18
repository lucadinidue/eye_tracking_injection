import sys
import os
sys.path.append(os.path.abspath('.'))
import torch
os.environ["CUDA_VISIBLE_DEVICES"] = "0" 

from modules.data.dataset_utils import create_senteces_from_data, scale_datasets, tokenize_and_align_labels
from modules.modeling.custom_modeling_roberta import RobertaForMultiTaskTokenClassification, RobertaForMultiTaskTokenClassificationWithWeight
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
    parser.add_argument('-b', '--batch_size', type=int, default=16)
    parser.add_argument('-l', '--learning_rate', dest='learning_rate', type=float, default=1e-05)
    parser.add_argument('-e', '--epochs', dest='training_epochs', type=int, default=50)
    parser.add_argument('-d', '--weight_decay', dest='weight_decay', type=float, default=0.0)
    parser.add_argument('-w', '--weighted_loss', dest='weighted_loss', action='store_true')
    args = parser.parse_args()

    set_seed(SEED)

    model_string = args.model_name.split('/')[-1]
    train_path = f'data/geco/dataset/pp{args.user_id}_dataset_train.csv'
    test_path = f'data/geco/dataset/pp{args.user_id}_dataset_test.csv'

    loss_dir = 'weighted_loss' if args.weighted_loss else 'average_loss'
    model_out_dir = f'models/eye_gaze_finetuning/{args.training_epochs}epochs_lr{args.learning_rate}_nowd/{model_string}_pp{args.user_id}'

    print(model_out_dir)

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
    
    num_epoch_steps = len(tokenized_train_dataset)/args.batch_size if len(tokenized_train_dataset) % args.batch_size == 0 else int(len(tokenized_train_dataset)/args.batch_size) + 1

    
    data_collator = DataCollatorForMultiTaskTokenClassification(tokenizer)
   
    mae = evaluate.load('mae')
    spearmanr = evaluate.load("spearmanr")
    def compute_metrics(eval_pred):
        res = dict()
        for task_idx, task in enumerate(trainer.label_names):
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

    
    config = AutoConfig.from_pretrained(args.model_name)
    config.update({'tasks': TASKS, 'keys_to_ignore_at_inference':['mse_loss', 'mae_loss', 'labels']})

#    if args.weighted_loss:
#        model = RobertaForMultiTaskTokenClassificationWithWeight.from_pretrained(args.model_name, config=config)
#    else:
    model = RobertaForMultiTaskTokenClassification.from_pretrained(args.model_name, config=config)
    

    training_args = TrainingArguments(
        output_dir=model_out_dir, 
        eval_strategy='epoch',
        logging_strategy='epoch',
        logging_dir=model_out_dir,
        label_names=[f'label_{task}' for task in TASKS],
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.training_epochs,
        save_steps=num_epoch_steps*10,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        save_strategy = 'no' #'steps' if args.training_epochs > 10 else 'no'
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
    trainer.save_model(model_out_dir)
    trainer.save_state()


if __name__ == '__main__':
    main()