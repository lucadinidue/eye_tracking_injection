import seaborn as sns
import pandas as pd
import argparse
import json
import os
import re
from IPython.display import display, HTML
pd.set_option('display.max_rows', None)

labels_map = {
    'full_model': 'ALL',
    'last_3': 'L3',
    'last_2': 'L2',
    'regressor_only': 'REGR',
    'classifier_only': 'CLF',
    'lora': 'LORA',
    'interleaved_multitask': 'IL-M',
    'silver_labels': 'SILV',
    'complexity_only': 'BL',
    'sentiment_only': 'BL'
}

metrics_map = {
    'sentiment': ['accuracy'],
    'complexity': ['mae', 'spearmanr'],
    'cola': ['matthews_correlation'],
    'mnli':['mismatched_accuracy', 'matched_accuracy'],
    'mrpc': ['accuracy', 'combined_score'],
    'qnli': ['accuracy'],
    'qqp': ['accuracy', 'combined_score'],
    'rte': ['accuracy'],
    'stsb': ['combined_score', 'pearson', 'spearmanr']
}

sns.set_style('darkgrid')

def load_trainer_state(model_dir):
    """Load trainer state from the specified directory."""
    trainer_state_path = os.path.join(model_dir, 'trainer_state.json')
    with open(trainer_state_path, 'r') as src_file:
        return json.load(src_file)

def parse_log_history_eval_metrics(log_history, task):
    """Parse evaluation metrics from the log history."""
    if task == 'complexity':
        logs_dict = parse_log_history_complexity(log_history)
    elif task == 'sentiment':
        logs_dict = parse_log_history_sentiment(log_history)
    else:
        logs_dict = parse_log_history_glue(log_history, task)
    return pd.DataFrame.from_dict(logs_dict)

def parse_log_history_glue(log_history, task):
    logs_dict = {'epoch': [], 'metric': [], 'score': []}
    metrics = metrics_map[task]
    for entry in log_history:
        for metric in metrics:
            for eval_entry in [f'eval_{metric}', f'eval_dst_{metric}', f'eval_validation_{metric}']:
                if eval_entry in entry:
                    logs_dict['epoch'].append(entry['epoch'])
                    logs_dict['metric'].append(metric)
                    logs_dict['score'].append(entry[eval_entry])
    return logs_dict

def parse_log_history_sentiment(log_history):
    logs_dict = {'epoch': [], 'metric': [], 'score': []}
    for entry in log_history:
        for eval_entry in ['eval_accuracy', 'eval_dst_accuracy']:
            if eval_entry in entry:
                logs_dict['epoch'].append(entry['epoch'])
                logs_dict['metric'].append('accuracy')
                logs_dict['score'].append(entry[eval_entry])
        if 'eval_sentiment' in entry:
            logs_dict['epoch'].append(entry['epoch'])
            logs_dict['metric'].append('accuracy')
            logs_dict['score'].append(entry['eval_sentiment']['accuracy'])
    return logs_dict

def parse_log_history_complexity(log_history):
    logs_dict = {'epoch': [], 'metric': [], 'score': []}
    for entry in log_history:
        if 'eval_complexity' in entry:
            for metric in ['mae', 'spearmanr']:
                logs_dict['epoch'].append(entry['epoch'])
                logs_dict['metric'].append(metric)
                logs_dict['score'].append(entry['eval_complexity'][metric])
        elif 'eval_loss' in entry or 'eval_dst_loss' in entry:
            prefix = 'eval_' if 'eval_loss' in entry else 'eval_dst_'
            for metric in ['mae', 'spearmanr']:
                logs_dict['epoch'].append(entry['epoch'])
                logs_dict['metric'].append(metric)
                logs_dict['score'].append(entry[f'{prefix}{metric}'])

    return logs_dict

def load_and_parse_trainer_state(model_dir, task):
    """Load trainer state and parse metrics for a given model directory."""
    trainer_state = load_trainer_state(model_dir)
    try:
        user_id = re.findall(r'pp(\d+)', model_dir)[0]
    except:
        user_id = 'no_ft'
    metrics_df = parse_log_history_eval_metrics(trainer_state['log_history'], task)
    metrics_df['user'] = user_id
    return metrics_df


def load_baseline_scores(src_path, task, metrics):
    trainer_state = load_trainer_state(src_path)
    metrics_df = parse_log_history_eval_metrics(trainer_state['log_history'], task)
    last_epoch_scores = metrics_df[metrics_df['epoch'] == metrics_df['epoch'].max()]
    try:
        metrics = {metric: last_epoch_scores[last_epoch_scores['metric'] == metric]['score'].item() for metric in metrics}
    except:
        metrics = {metric: last_epoch_scores[last_epoch_scores['metric'] == f'eval_{metric}']['score'].item() for metric in metrics}
    return metrics

def load_metrics_dataframe(src_dir, task):
    metrics_dfs = []
    for finetuning_config in os.listdir(src_dir):
        config_path = os.path.join(src_dir, finetuning_config)
        if not 'trainer_state.json' in os.listdir(config_path):
            for user_dir_name in os.listdir(config_path):
                if finetuning_config == 'lora' and 'adapters' not in user_dir_name:
                    continue
                user_path = os.path.join(config_path, user_dir_name)
                try:
                    metrics_df = load_and_parse_trainer_state(user_path, task)
                except:
                    continue
                metrics_df['model'] = finetuning_config
                last_epoch_df = metrics_df[metrics_df['epoch'] == metrics_df['epoch'].max()]
                metrics_dfs.append(last_epoch_df)     
    metrics_dfs = pd.concat(metrics_dfs, axis=0)
    return metrics_dfs

def plot_metrics(df, metric, output_path):
    plot = sns.heatmap(data=df, annot=True, cmap='crest_r' if metric == 'mae' else 'crest', cbar=False);
    plot.set_yticklabels(plot.get_yticklabels(), rotation=0)
    # plot.set_xticklabels(plot.get_xticklabels(), rotation=-45)
    plot.vlines([7], *plot.get_ylim(), color='white');
    plot.set_title(metric.upper())
    plot.figure.tight_layout()
    plot.figure.savefig(output_path)
    plot.cla()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--downstream_task', type=str)
    args = parser.parse_args()

    models_directory = f'models/{args.downstream_task}/10epochs_lr1e-05'
    plots_out_template = f'results/downstream_tasks/{args.downstream_task}'
    baseline_dir = f'models/{args.downstream_task}/roberta-base_baseline'
    
    metrics  = metrics_map[args.downstream_task] 

    baseline_metrics = load_baseline_scores(baseline_dir, args.downstream_task, metrics)
    metrics_df = load_metrics_dataframe(models_directory, args.downstream_task)

    # metrics_df = metrics_df[['user', 'score', 'model', 'metric']]

    for metric in metrics:
        metric_df = metrics_df[metrics_df['metric'] == metric]
        pivoted_df = metric_df.pivot(index='user', columns='model', values='score')
        pivoted_df[f'{args.downstream_task}_only'] = baseline_metrics[metric]
        clf_or_regressor_str = 'regressor_only' if 'regressor_only' in pivoted_df else 'classifier_only'
        pivoted_df = pivoted_df.reindex(['full_model', 'last_3', 'last_2', clf_or_regressor_str, 'lora', 'interleaved_multitask', 'silver_labels', f'{args.downstream_task}_only'], axis=1)
        pivoted_df = pivoted_df.rename(columns=labels_map)
        out_path = plots_out_template + f'_{metric}.png'

        
        plot_metrics(pivoted_df, metric, out_path)
    

    
    

if __name__ == '__main__':
    main()
