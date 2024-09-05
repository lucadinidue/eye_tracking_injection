import sys
import os
sys.path.append(os.path.abspath('.'))

from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding, TrainingArguments, Trainer
from complexity_finetuning import load_complexity_dataset
import evaluate
import argparse

def sum_attention_weights(eye_tracking_model_path, downstream_task_model_path, output_path):
    eye_tracking_model = AutoModelForSequenceClassification.from_pretrained(eye_tracking_model_path)
    downstream_task_model = AutoModelForSequenceClassification.from_pretrained(downstream_task_model_path)

    eye_tracking_state_dict = eye_tracking_model.state_dict()
    downstream_task_state_dict = downstream_task_model.state_dict()

    for key in eye_tracking_state_dict.keys():
        # if 'attention' in key:
        #     if 'query' in key or 'value' in key or 'key' in key:
        if 'classifier' not in key:
            downstream_task_state_dict[key] = downstream_task_state_dict[key] + eye_tracking_state_dict[key]
    
    downstream_task_model.load_state_dict(downstream_task_state_dict)
    # downstream_task_model.save_pretrained(output_path)
    return downstream_task_model

def load_and_preprocess_complexity_dataset(test_path, tokenizer):
    test_dataset = load_complexity_dataset(test_path)
    
    def preprocess_function(examples):
        return tokenizer(examples['text'], truncation=True)

    tokenized_test_dataset = test_dataset.map(preprocess_function, desc="Running tokenizer on dataset")
    return tokenized_test_dataset

def evaluate_model(model, tokenized_test_dataset, data_collator):
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
    
    training_args = TrainingArguments(
        do_train=False,
        do_eval=True,
        output_dir='prova',
        eval_strategy='epoch',
        per_device_eval_batch_size=16,
        save_strategy='no' 
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        eval_dataset=tokenized_test_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    
    res = trainer.evaluate()
    return res

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--downstream_task', type=str, choices=['sentiment', 'complexity'])
    parser.add_argument('-o', '--tokenizer_name', default='FacebookAI/roberta-base')
    args = parser.parse_args()

    eye_tracking_models_dir = 'models/eye_gaze_finetuning/average_loss/50epochs_lr1e-05'
    downstream_task_models_dir = f'models/{args.downstream_task}/50epochs_lr1e-05'
    simple_finetuning_model_path = 'models/complexity/roberta-base'

   
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, add_prefix_space=True)
    if args.downstream_task == 'complexity':
        test_path = 'data/complexity/complexity_ds_en_test.csv'
        tokenized_test_dataset = load_and_preprocess_complexity_dataset(test_path, tokenizer)
    else:
        raise Exception(f'Task {args.downstream_task} not implemented yet.')
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)


    for user_dir in os.listdir(eye_tracking_models_dir):
        eye_tracking_model_path = os.path.join(eye_tracking_models_dir, user_dir)
        output_path = os.path.join(downstream_task_models_dir, user_dir+'_task_vectors')        
        downstream_task_model = sum_attention_weights(eye_tracking_model_path, simple_finetuning_model_path, output_path)
        print(user_dir)
        res = evaluate_model(downstream_task_model, tokenized_test_dataset, data_collator)
        print(res)
        print('\n\n')

if __name__ == '__main__':
    main()
        

