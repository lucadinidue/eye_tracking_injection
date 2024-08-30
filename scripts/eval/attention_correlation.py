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


def compute_average_user_correlation(all_correlation_dfs, grouping_variable='epoch'):
    mean_correlation_row = all_correlation_dfs[all_correlation_dfs['user'] != 'no'].groupby([grouping_variable, 'layer'])['correlation'].mean().reset_index()
    mean_correlation_row['user'] = 'ft_avg'
    return mean_correlation_row

def extract_layers_correlations(model_dir, eye_tracking_data, positive_correlation, all_correlations_dict, grouping_value, user_id, grouping_variable='epoch'):
    for layer in range(12):
        layer_attention_path = os.path.join(model_dir, f'{layer}.json')
        layer_attention = load_model_attention(layer_attention_path)
        corr = compute_correlation(eye_tracking_data, layer_attention, positive_correlation)
        all_correlations_dict['layer'].append(layer)
        all_correlations_dict['correlation'].append(corr)
        all_correlations_dict[grouping_variable].append(grouping_value)
        all_correlations_dict['user'].append(user_id)


def attention_correlation_base(eye_tracking_dir, src_attention_dir, grouping_variable, args):
    model_attention_dir = args.model_input_directory

    all_correlation_dict = {'layer': [], 'correlation': [], 'user': [], 'epoch':[]}
    for model_dir_name in os.listdir(model_attention_dir):
        user_id = re.findall(r'pp(\d*)$', model_dir_name)[0]
        eye_tracking_path = os.path.join(eye_tracking_dir, f'pp{user_id}_dataset_test.csv')
        eye_tracking_data = load_eye_tracking_data(eye_tracking_path, args.eye_tracking_feature)
        for checkpoint_dir_name in os.listdir(os.path.join(model_attention_dir, model_dir_name)):
            checkpoint_dir = os.path.join(model_attention_dir, model_dir_name, checkpoint_dir_name) 
            epoch = int(checkpoint_dir.split('-')[-1]) * 10           
            extract_layers_correlations(checkpoint_dir, eye_tracking_data, args.positive_correlation, all_correlation_dict, epoch, user_id, grouping_variable)

    baseline_attention_dir = os.path.join(src_attention_dir, 'base', 'roberta-base')
    for epoch in list(set(all_correlation_dict['epoch'])):
        extract_layers_correlations(baseline_attention_dir, eye_tracking_data, args.positive_correlation, all_correlation_dict, epoch, 'no_ft', grouping_variable)
    all_correlations_df = pd.DataFrame.from_dict(all_correlation_dict)
    
    return all_correlations_df



def attention_correlation_downstream_tasks(eye_tracking_dir, src_attention_dir, grouping_variable, args):
    model_attention_dir = args.model_input_directory

    all_correlation_dict = {'layer': [], 'correlation': [], 'user': [], 'trainable': []}
    for trainable_config in os.listdir(model_attention_dir):
        trainable_dir = os.path.join(model_attention_dir, trainable_config)
        for model_dir_name in os.listdir(trainable_dir):
            model_dir = os.path.join(trainable_dir, model_dir_name)
            user_id = re.findall(r'pp(\d*)$', model_dir_name)[0]
            eye_tracking_path = os.path.join(eye_tracking_dir, f'pp{user_id}_dataset_test.csv')
            eye_tracking_data = load_eye_tracking_data(eye_tracking_path, args.eye_tracking_feature)
            extract_layers_correlations(model_dir, eye_tracking_data, args.positive_correlation, all_correlation_dict, trainable_config, user_id, grouping_variable)

    baseline_attention_dir = os.path.join(src_attention_dir, args.downstream_task, 'roberta-base')
    for trainable_config in list(set(all_correlation_dict['trainable'])):
        extract_layers_correlations(baseline_attention_dir, eye_tracking_data, args.positive_correlation, all_correlation_dict, trainable_config, 'no_ft', grouping_variable)
    all_correlations_df = pd.DataFrame.from_dict(all_correlation_dict)

    return all_correlations_df



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--eye_tracking_feature', type=str, default='dur', help='Eye-tracking feature used to compute correlation.')
    parser.add_argument('-p', '--positive_correlation', type=bool, default=True, help='Computes the absolute value of correlation coefficients.')
    parser.add_argument('-i', '--model_input_directory', type=str, help='Model configuration.')
    parser.add_argument('-t', '--downstream_task', type=str, default='base', choices=['base', 'complexity', 'sentiment', 'interleaved_complexity'], help='Indicates the downstream task on which the model has been finetuned.')
    args = parser.parse_args()

    eye_tracking_dir = 'data/geco/dataset/'
    model_attention_dir = args.model_input_directory
    plots_dir = f'results/attention_correlations/{args.downstream_task}'

    if args.downstream_task == 'base' or 'interleaved' in args.downstream_task:
        grouping_variable = 'epoch'
        all_correlations_df = attention_correlation_base(eye_tracking_dir, model_attention_dir, grouping_variable, args)
    else:
        grouping_variable = 'trainable'
        all_correlations_df = attention_correlation_downstream_tasks(eye_tracking_dir, model_attention_dir, grouping_variable, args)
    
    # Add average user correlation
    mean_correlation_row = compute_average_user_correlation(all_correlations_df, grouping_variable)
    all_correlations_df = pd.concat([all_correlations_df, mean_correlation_row], ignore_index=True)


    plot_correlations(all_correlations_df, plots_dir, args.downstream_task, grouping_variable)


if __name__ == '__main__':
    main()
