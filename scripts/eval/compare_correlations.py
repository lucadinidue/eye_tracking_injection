import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import argparse
import json
import os

sns.set_style('darkgrid')


def load_json(src_path):
    """Load model attention data from a JSON file."""
    with open(src_path, 'r') as src_file:
        return json.load(src_file)


def plot_correlations(correlation_df, plots_dir, out_dir, grouping_variable='epoch'):
    """Plot correlation heatmaps across epochs or other grouping variables and save the plot."""
    vmin, vmax = correlation_df['correlation'].min(), correlation_df['correlation'].max()
    var_values = sorted(correlation_df[grouping_variable].unique())
    y_size = len(var_values) * 6
    fig, axes = plt.subplots(len(var_values), 1, sharex=True, figsize=(10, y_size))
    
    if len(var_values) == 1:
        axes = [axes]  # Ensure axes is always a list for consistency

    for idx, value in enumerate(var_values):
        pivoted_df = correlation_df[correlation_df[grouping_variable] == value].pivot(index='user', columns='layer', values='correlation')
        sns.heatmap(data=pivoted_df, annot=True, cmap='crest', cbar=False, ax=axes[idx], vmin=vmin, vmax=vmax)
        axes[idx].set_title(f'{grouping_variable} {value}')
        axes[idx].set_yticklabels(axes[idx].get_yticklabels(), rotation=0)

    plt.tight_layout()
    fig.savefig(os.path.join(plots_dir, f'roberta_base_{out_dir}.png'))

def convert_dict_to_df(correlations_dict):
    df_dict = {'model':[], 'user':[], 'layer':[], 'score':[]}
    for model, model_correlations in correlations_dict.items():
        for user, user_correlations in model_correlations.items():
            for layer, score in user_correlations.items():
                df_dict['model'].append(model)
                df_dict['user'].append(user)
                df_dict['layer'].append(int(layer))
                df_dict['score'].append(score)
    return pd.DataFrame.from_dict(df_dict)


def load_trainer_state(model_dir):
    """Load trainer state from the specified directory."""
    trainer_state_path = os.path.join(model_dir, 'trainer_state.json')
    with open(trainer_state_path, 'r') as src_file:
        return json.load(src_file)

def get_last_epoch_eval_metrics(model_dir):
    log_history = load_trainer_state(model_dir)['log_history']
    metrics = {}
    for entry in log_history:
        if 'eval_loss' in entry or 'eval_dst_loss' in entry:
            prefix = 'eval_' if 'eval_loss' in entry else 'eval_dst_'
            for metric in ['mae', 'spearmanr']:
                try:
                    metrics[metric] = entry[f'{prefix}{metric}']
                except:
                     metrics[metric] = entry['eval_complexity'][metric]
    return metrics



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--downstream_task', type=str, default='base', choices=['complexity', 'sentiment'], help='Indicates the downstream task on which the model has been finetuned.')
    args = parser.parse_args()

    json_correlations_path = f'attentions/all_correlations_{args.downstream_task}.json'
    output_path = f'results/attention_correlations/{args.downstream_task}_comparison.png'

    correlations_dict = load_json(json_correlations_path)
    correlations_df = convert_dict_to_df(correlations_dict)

    user_ids = correlations_df['user'].unique().tolist()
    vmin, vmax = correlations_df['score'].min(), correlations_df['score'].max()
    fig, axes = plt.subplots(len(user_ids)-1, 1, sharex=True, figsize=(10, 60))

    only_complexity_df = correlations_df[correlations_df['user']=='no']

    for idx, user in enumerate(sorted(user_ids)):
        user_df = correlations_df[correlations_df['user']==user]
        if user != 'no': #'base' in user_df[user_df['user'] == user]['model'].tolist():
            user_df = pd.concat([user_df, only_complexity_df], axis=0)

            pivoted_df = user_df.pivot(index='model', columns='layer', values='score')
            pivoted_df = pivoted_df.reindex(['base', 'complexity_full', 'complexity_last_3', 'complexity_last_2', 'complexity_lora', 'complexity_interleaved', 'complexity_silver_labels', 'complexity_only'])
            pivoted_df['avg'] = pivoted_df.mean(axis=1)

            plot = sns.heatmap(pivoted_df, annot=True, cmap='crest', cbar=False, ax=axes[idx], vmin=vmin, vmax=vmax)
            plot.set_title(f'User {user}')
            plot.hlines([1, 7], *plot.get_xlim(), color='white')
            plot.vlines([12], *plot.get_ylim(), color='white')

    plt.tight_layout()
    fig.savefig(output_path)

if __name__ == '__main__':
    main()
