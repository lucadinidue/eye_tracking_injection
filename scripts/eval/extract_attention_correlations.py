import os
import sys
sys.path.append(os.path.abspath(".")) 

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from modules.data.dataset_utils import create_senteces_from_data
from modules.data.attention_utils import save_dictionary
from scipy.stats import spearmanr
import seaborn as sns
import pandas as pd
import numpy as np
import argparse
import json
import re

sns.set_style('darkgrid')


def load_eye_tracking_data(src_path, col):
    """Load and preprocess eye-tracking data."""
    data = pd.read_csv(src_path, index_col=0)
    gaze_dataset = create_senteces_from_data(data, [col], keep_id=True)
    return gaze_dataset


def load_json(src_path):
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

def compute_attention_correlation(model_dir, eye_tracking_data, positive_correlation):
    correlations_dict = {}
    for layer in range(12):
        layer_attention_path = os.path.join(model_dir, f'{layer}.json')
        layer_attention = load_json(layer_attention_path)
        corr = compute_correlation(eye_tracking_data, layer_attention, positive_correlation)
        correlations_dict[layer] = corr
    return correlations_dict

def compute_attention_correlation_checkpoints(model_dir, eye_tracking_data, positive_correlation):
    correlations_dict = {}
    for checkpoint_name in os.listdir(model_dir):
        checkpoint_dir = os.path.join(model_dir, checkpoint_name)
        checkpoint_num = int(checkpoint_name.split('-')[-1])
        checkpoint_correlations = compute_attention_correlation(checkpoint_dir, eye_tracking_data, positive_correlation)
        correlations_dict[checkpoint_num] = checkpoint_correlations
    return correlations_dict

def find_last_checkpoint(model_dir):
    dir_content = os.listdir(model_dir)
    if dir_content[0].startswith('epoch'):
        checkpoints = [int(el.split('_')[-1]) for el in dir_content]
        last_checkpoint = sorted(checkpoints)[-1]
        return os.path.join(model_dir, f'epoch_{last_checkpoint}')
    else:
        return model_dir


def init_correlations_dict(all_correlations_json_path, model_string):
    if os.path.exists(all_correlations_json_path):
        all_correlations_dict = load_json(all_correlations_json_path)
    else:
        all_correlations_dict = {}
    all_correlations_dict[model_string] =  {}
    return all_correlations_dict
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--eye_tracking_feature', type=str, default='dur', help='Eye-tracking feature used to compute correlation.')
    parser.add_argument('-p', '--positive_correlation', type=bool, default=True, help='Computes the absolute value of correlation coefficients.')
    parser.add_argument('-i', '--models_input_directory', type=str, help='Directory from where to load models\' attentions.')
    parser.add_argument('-s', '--model_string', type=str, help='The string to use to identify the models\' correlations.')
    parser.add_argument('-o', '--output_path', type=str, help='The json file where to save the extracted attentions.')
    parser.add_argument('-c', '--all_checkpoints', action='store_true')
    args = parser.parse_args()

    eye_tracking_dir = 'data/geco/dataset/'
     
    all_correlations_dict = init_correlations_dict(args.output_path, args.model_string)

    if '1.json' in os.listdir(args.models_input_directory): # model not finetuned on eye-tracking
        eye_tracking_path = os.path.join(eye_tracking_dir, f'pp21_dataset_test.csv')
        eye_tracking_data = load_eye_tracking_data(eye_tracking_path, args.eye_tracking_feature)
        correlation_dict = compute_attention_correlation(args.models_input_directory, eye_tracking_data, args.positive_correlation)
        user_id = 'no'
        all_correlations_dict[args.model_string][user_id] = correlation_dict
    else:
        for model_name in os.listdir(args.models_input_directory):
            model_dir = os.path.join(args.models_input_directory, model_name)
            user_id = int(re.findall(r'pp(\d*)$', model_name)[0])
            eye_tracking_path = os.path.join(eye_tracking_dir, f'pp{user_id}_dataset_test.csv')
            eye_tracking_data = load_eye_tracking_data(eye_tracking_path, args.eye_tracking_feature)
            if not args.all_checkpoints:
                model_dir = find_last_checkpoint(model_dir)
                correlation_dict = compute_attention_correlation(model_dir, eye_tracking_data, args.positive_correlation)
            else: 
                correlation_dict = compute_attention_correlation_checkpoints(model_dir, eye_tracking_data, args.positive_correlation)  
    
            all_correlations_dict[args.model_string][user_id] = correlation_dict
    save_dictionary(all_correlations_dict, args.output_path)    
    
    
if __name__ == '__main__':
    main()
