import seaborn as sns
import pandas as pd
import argparse
import json
import os
import re

sns.set_style('darkgrid')

def load_trainer_state(model_dir):
    """Load trainer state from the specified directory."""
    trainer_state_path = os.path.join(model_dir, 'trainer_state.json')
    with open(trainer_state_path, 'r') as src_file:
        return json.load(src_file)

def parse_log_history_eval_metrics(log_history):
    """Parse evaluation metrics from the log history."""
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
    return pd.DataFrame.from_dict(logs_dict)

def load_and_parse_trainer_state(model_dir):
    """Load trainer state and parse metrics for a given model directory."""
    trainer_state = load_trainer_state(model_dir)
    try:
        user_id = re.findall(r'pp(\d+)', model_dir)[0]
    except:
        user_id = 'no_ft'
    metrics_df = parse_log_history_eval_metrics(trainer_state['log_history'])
    metrics_df['user'] = user_id
    return metrics_df


def load_baseline_scores(src_path):
    trainer_state = load_trainer_state(src_path)
    metrics_df = parse_log_history_eval_metrics(trainer_state['log_history'])
    last_epoch_scores = metrics_df[metrics_df['epoch'] == metrics_df['epoch'].max()]
    metrics = {
        'mae': last_epoch_scores[last_epoch_scores['metric'] == 'mae']['score'].item(),
        'spearmanr': last_epoch_scores[last_epoch_scores['metric'] == 'spearmanr']['score'].item()
    }
    return metrics

def load_metrics_dataframe(src_dir):
    metrics_dfs = []
    for finetuning_config in os.listdir(src_dir):
        config_path = os.path.join(src_dir, finetuning_config)
        if not 'trainer_state.json' in os.listdir(config_path):
            for user_dir_name in os.listdir(config_path):
                if finetuning_config == 'lora' and 'adapters' not in user_dir_name:
                    continue
                user_path = os.path.join(config_path, user_dir_name)
                metrics_df = load_and_parse_trainer_state(user_path)
                metrics_df['model'] = finetuning_config
                last_epoch_df = metrics_df[metrics_df['epoch'] == metrics_df['epoch'].max()]
                metrics_dfs.append(last_epoch_df)
    metrics_dfs = pd.concat(metrics_dfs, axis=0)
    return metrics_dfs

def plot_metrics(df, metric, output_path):
    plot = sns.heatmap(data=df, annot=True, cmap='crest' if metric == 'spearmanr' else 'crest_r', cbar=False);
    plot.vlines([7], *plot.get_ylim(), color='white');
    plot.set_title(metric.upper())
    plot.figure.tight_layout()
    plot.figure.savefig(output_path)
    plot.cla()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--downstream_task', type=str)
    args = parser.parse_args()

    models_directory = f'models/{args.downstream_task}'
    plots_out_dir = f'results/{args.downstream_task}'
    baseline_dir = f'models/{args.downstream_task}/roberta-base_baseline'
    
    baseline_metrics = load_baseline_scores(baseline_dir)
    metrics_df = load_metrics_dataframe(models_directory)

    for metric in ['mae', 'spearmanr']:
        metric_df = metrics_df[metrics_df['metric'] == metric]
        pivoted_df = metric_df.pivot(index='user', columns='model', values='score')
        pivoted_df['complexity_only'] = baseline_metrics[metric]
        pivoted_df = pivoted_df.reindex(['full_model', 'last_3', 'last_2', 'regressor_only', 'lora', 'interleaved_multitask', 'silver_labels', 'complexity_only'], axis=1)
        plot_metrics(pivoted_df, metric, f'{plots_out_dir}/{metric}_comparison.png')
    
    

if __name__ == '__main__':
    main()
