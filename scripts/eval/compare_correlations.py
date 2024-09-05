import os

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import os
import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

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
                df_dict['user'].append(int(user))
                df_dict['layer'].append(int(layer))
                df_dict['score'].append(score)
    return pd.DataFrame.from_dict(df_dict)


def main():
    json_correlations_path = 'attentions/all_correlations.json'
    output_path = 'results/attention_correlations/attention_correlation_comparison.png'
    
    correlations_dict = load_json(json_correlations_path)
    correlations_df = convert_dict_to_df(correlations_dict)

    grouped_df = correlations_df[['model', 'layer', 'score']].groupby(['model', 'layer']).mean().reset_index()
    pivoted_df = grouped_df.pivot(index='model', columns='layer', values='score')
    sns.heatmap(pivoted_df, annot=True, cmap='crest', cbar=False)

    # p = sns.lineplot(data=correlations_df, x='layer', y='score', hue='model')
    # p.figure.savefig(output_path)

    # plot_correlations(all_correlations_df, plots_dir, args.downstream_task, grouping_variable)


if __name__ == '__main__':
    main()
