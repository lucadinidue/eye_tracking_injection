
import os
import sys
sys.path.append(os.path.abspath(".")) 

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import os
import json
import re
import argparse
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from modules.data.dataset_utils import create_senteces_from_data

sns.set_style('darkgrid')


def load_eye_tracking_data(src_path, col):
    """Load and preprocess eye-tracking data."""
    data = pd.read_csv(src_path, index_col=0)
    gaze_dataset = create_senteces_from_data(data, [col], keep_id=True)
    return gaze_dataset


def load_model_attention(src_path):
    """Load model attention data from a JSON file."""
    with open(src_path, 'r') as src_file:
        return json.load(src_file)


def normalize_list(l):
    """Normalize a list of values to the range [0, 1]."""
    l = np.array(l)
    return ((l - l.min()) / (l.max() - l.min())).tolist()


def compute_correlation(eye_tracking_dataset, model_attention_dict, positive_corr=True):
    """Compute the Spearman correlation between eye-tracking and model attention data."""
    all_human_attentions = []
    all_model_attentions = []

    feature_name = [col for col in eye_tracking_dataset.column_names if col not in ['id', 'text']][0]

    for sent_id, human_attention in zip(eye_tracking_dataset['id'], eye_tracking_dataset[feature_name]):
        if sent_id in model_attention_dict:
            model_attention = model_attention_dict[sent_id]
            all_human_attentions += normalize_list(human_attention)
            all_model_attentions += normalize_list(model_attention)

    s = spearmanr(all_model_attentions, all_human_attentions)
    if s.pvalue < 0.05:
        return max(s.statistic, -s.statistic) if positive_corr else s.statistic
    return None


def compute_correlation_df_across_epochs(model_attention_dir, eye_tracking_data, positive_corr=True):
    """Compute correlations across epochs for each layer."""
    correlations = {'epoch': [], 'layer': [], 'correlation': []}

    for checkpoint_dir in os.listdir(model_attention_dir):
        checkpoint_path = os.path.join(model_attention_dir, checkpoint_dir)
        epoch = int(checkpoint_dir.split('-')[-1]) * 10

        for layer in range(12):
            layer_attention_path = os.path.join(checkpoint_path, f'{layer}.json')
            layer_attention = load_model_attention(layer_attention_path)
            corr = compute_correlation(eye_tracking_data, layer_attention, positive_corr)

            correlations['epoch'].append(epoch)
            correlations['layer'].append(layer)
            correlations['correlation'].append(corr)

    return pd.DataFrame(correlations)


def compute_baseline_correlation(user_attention_dir, eye_tracking_data, epochs, positive_corr=True):
    """Compute baseline correlations across all epochs for a user."""
    correlations = {'epoch': [], 'layer': [], 'correlation': []}

    for layer in range(12):
        layer_attention_path = os.path.join(user_attention_dir, f'{layer}.json')
        layer_attention = load_model_attention(layer_attention_path)
        corr = compute_correlation(eye_tracking_data, layer_attention, positive_corr)

        for epoch in epochs:
            correlations['epoch'].append(epoch)
            correlations['layer'].append(layer)
            correlations['correlation'].append(corr)

    return pd.DataFrame(correlations)


def plot_correlations(all_correlation_dfs, plots_dir, model_config):
    """Plot correlation heatmaps across epochs and save the plot."""
    vmin, vmax = all_correlation_dfs['correlation'].min(), all_correlation_dfs['correlation'].max()
    epochs = sorted(all_correlation_dfs['epoch'].unique())

    # fig, axes = plt.subplots(len(epochs), 1, sharex=True, figsize=(10, 10))
    fig, axes = plt.subplots(len(epochs), 1, sharex=True, figsize=(7, 30))
    
    for idx, epoch in enumerate(epochs):
        epoch_correlations_df = all_correlation_dfs[all_correlation_dfs['epoch'] == epoch]
        pivoted_df = epoch_correlations_df.pivot(index='user', columns='layer', values='correlation')

        if len(epochs) > 1:
            sns.heatmap(data=pivoted_df, annot=True, cmap='crest', cbar=False, ax=axes[idx], vmin=vmin, vmax=vmax)
            axes[idx].set_title(f'Epoch {epoch}')
            axes[idx].set_yticklabels(axes[idx].get_yticklabels(), rotation=0)
        else:
            sns.heatmap(data=pivoted_df, annot=True, cmap='crest', cbar=False, ax=axes, vmin=vmin, vmax=vmax)
            axes.set_title(f'Epoch {epoch}')
            axes.set_yticklabels(axes.get_yticklabels(), rotation=0)

    plt.tight_layout()
    fig.savefig(os.path.join(plots_dir, f'roberta_base_{model_config}.png'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--eye_tracking_feature', type=str, default='dur', help='Feature to use from eye-tracking data')
    parser.add_argument('-p', '--positive_correlation', type=bool, default=True, help='Consider only positive correlations')
    parser.add_argument('-c', '--model_config', type=str, default='10epochs_lr5e-05', help='Model configuration')
    args = parser.parse_args()

    eye_tracking_dir = 'data/geco/dataset/'
    model_attention_dir = 'attentions/'
    plots_dir = 'results/attention_correlations'

    all_correlation_dfs = []

    for file_name in os.listdir(eye_tracking_dir):
        if 'test' in file_name:
            user_id = re.findall(r'pp(\d*)_', file_name)[0]
            user_attention_dir = os.path.join(model_attention_dir, args.model_config, f'roberta-base_pp{user_id}')
            eye_tracking_path = os.path.join(eye_tracking_dir, file_name)
            eye_tracking_data = load_eye_tracking_data(eye_tracking_path, args.eye_tracking_feature)

            correlations_df = compute_correlation_df_across_epochs(user_attention_dir, eye_tracking_data, args.positive_correlation)
            correlations_df['user'] = user_id
            all_correlation_dfs.append(correlations_df)

    # Add baseline correlation
    epochs = list(range(10, int(args.model_config[:2])+10, 10))
    baseline_attention_dir = os.path.join(model_attention_dir, 'roberta-base')
    baseline_eye_tracking_path = os.path.join(eye_tracking_dir, 'pp21_dataset_test.csv')
    baseline_eye_tracking_data = load_eye_tracking_data(baseline_eye_tracking_path, args.eye_tracking_feature)

    baseline_correlation_df = compute_baseline_correlation(baseline_attention_dir, baseline_eye_tracking_data, epochs, args.positive_correlation)
    baseline_correlation_df['user'] = 'no'
    all_correlation_dfs.append(baseline_correlation_df)

    # Combine all correlation dataframes
    all_correlation_dfs = pd.concat(all_correlation_dfs, axis=0)

    # Add average user correlation
    mean_correlation = all_correlation_dfs[all_correlation_dfs['user'] != 'no'].groupby(['epoch', 'layer'])['correlation'].mean().reset_index()
    mean_correlation['user'] = 'ft_avg'
    all_correlation_dfs = pd.concat([all_correlation_dfs, mean_correlation], ignore_index=True)

    # Plot the results
    plot_correlations(all_correlation_dfs, plots_dir, args.model_config)


if __name__ == '__main__':
    main()
