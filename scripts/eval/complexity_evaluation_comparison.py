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
        if 'eval_loss' in entry:
            for metric in ['mae', 'spearmanr']:
                logs_dict['epoch'].append(entry['epoch'])
                logs_dict['metric'].append(metric)
                logs_dict['score'].append(entry[f'eval_{metric}'])
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
    freeze = '_'.join(model_dir.split('/')[-1].split('_')[-2:])
    metrics_df['freeze'] = freeze if 'pp' not in freeze else 'full'
    if user_id == 'no_ft':
        metrics_df['freeze'] = 'no_finetuning'
    return metrics_df
    
def create_output_directory(directory):
    """Create output directory if it does not exist."""
    os.makedirs(directory, exist_ok=True)

def plot_metrics(metrics_dfs, plots_out_dir):
    """Generate and save line plots and heatmaps for the metrics."""
    hue_order = ['full', 'last_3', 'last_2', 'regressor_only', 'no_finetuning']

    # MAE across epochs
    plot_lineplot(metrics_dfs, 'mae', hue_order, os.path.join(plots_out_dir, 'mae_across_epochs.png'))

    # Spearman correlation across epochs
    plot_lineplot(metrics_dfs, 'spearmanr', hue_order, os.path.join(plots_out_dir, 'spearman_across_epochs.png'))

    # Heatmaps for the last epoch
    last_epoch_df = metrics_dfs[metrics_dfs['epoch'] == 10]
    plot_heatmap(last_epoch_df, 'mae', os.path.join(plots_out_dir, 'mae.png'))
    plot_heatmap(last_epoch_df, 'spearmanr', os.path.join(plots_out_dir, 'spearman.png'))

def plot_lineplot(metrics_dfs, metric, hue_order, output_path):
    """Plot and save line plot for a specific metric."""
    p = sns.lineplot(
        data=metrics_dfs[metrics_dfs['metric'] == metric],
        x='epoch', y='score', hue='freeze', hue_order=hue_order
    )
    p.set_title(metric.upper())
    p.figure.tight_layout()
    p.figure.savefig(output_path)
    p.cla()

def plot_heatmap(df, metric, output_path):
    """Plot and save heatmap for a specific metric."""
    pivoted_df = df[df['metric'] == metric].pivot(index='freeze', columns='user', values='score')
    p = sns.heatmap(data=pivoted_df, annot=True, cmap='crest' if metric == 'spearmanr' else 'crest_r', cbar=False)
    p.set_title(metric.upper())
    p.figure.tight_layout()
    p.figure.savefig(output_path)
    p.cla()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--models_input_directory', type=str, help='The models\' subdirectory')
    parser.add_argument('-o', '--output_directory', type=str, help='Output directory')
    args = parser.parse_args()

    plots_out_dir = f'results/complexity/{args.output_directory}'

    create_output_directory(plots_out_dir)

    model_dirs = [os.path.join(args.models_input_directory, model_name) for model_name in os.listdir(args.models_input_directory)]

    metrics_dfs = []
    for model_dir in model_dirs:
        print(model_dir)
        metrics_df = load_and_parse_trainer_state(model_dir)
        if not metrics_df.empty:
            metrics_dfs.append(metrics_df)

    # Load performance of the model not fine-tuned on eye-tracking data
    not_finetuned_dir = 'models/complexity/roberta-base'
    not_finetuned_metrics_df = load_and_parse_trainer_state(not_finetuned_dir)
    metrics_dfs.append(not_finetuned_metrics_df)

    # Combine all metrics dataframes
    metrics_dfs = pd.concat(metrics_dfs, axis=0)

    # Plot and save the metrics
    plot_metrics(metrics_dfs, plots_out_dir)

if __name__ == '__main__':
    main()
