import seaborn as sns
import pandas as pd
import argparse
import json
import os
import re
sns.set_style('darkgrid')

def parse_log_history_loss(log_history):
    logs_dict = {'epoch': [], 'data':[], 'loss':[]}
    for el in log_history:
        if 'loss' in el:
            logs_dict['epoch'].append(el['epoch'])
            logs_dict['data'].append('train')
            logs_dict['loss'].append(el['loss'])
        elif 'eval_loss' in el:
                logs_dict['epoch'].append(el['epoch'])
                logs_dict['data'].append('eval')
                logs_dict['loss'].append(el['eval_loss'])
        elif 'eval_eye_gaze_loss' in el:
            logs_dict['epoch'].append(el['epoch'])
            logs_dict['data'].append('eval_eye_gaze_loss')
            logs_dict['loss'].append(el['eval_eye_gaze_loss'])
        elif 'eval_dst_loss' in el:
            logs_dict['epoch'].append(el['epoch'])
            logs_dict['data'].append('eval_dst_loss')
            logs_dict['loss'].append(el['eval_dst_loss'])
    return pd.DataFrame.from_dict(logs_dict)


def parse_log_history_eval_metrics(log_history):
    logs_dict = {'epoch': [], 'metric':[], 'score':[], 'label':[]}
    for el in log_history:
        if any('eval_label_' in k for k in el.keys()):
            prefix = 'eval_label_'
            epoch = el['epoch']
            for k, v in el.items():
                if k.startswith(prefix):
                    label = k[len(prefix):]
                    for metric, score in v.items():
                        logs_dict['epoch'].append(epoch)
                        logs_dict['label'].append(label)
                        logs_dict['metric'].append(metric)
                        logs_dict['score'].append(score)
    return pd.DataFrame.from_dict(logs_dict)


def load_log_history_df(models_src_dir, loading_function):
    all_dfs = []
    for model_name in os.listdir(models_src_dir):
        user_id =  re.findall(r'pp(\d*)$', model_name)[0]
        trainer_state_path = os.path.join(models_src_dir, model_name, 'trainer_state.json')
        with open(trainer_state_path, 'r') as src_file:
            trainer_state = json.load(src_file)
        eval_metrics_df = loading_function(trainer_state['log_history'])
        eval_metrics_df['user'] = user_id
        all_dfs.append(eval_metrics_df)
    all_dfs = pd.concat(all_dfs, axis=0)
    return all_dfs

def save_lineplot(data, x, y, hue, title, out_path):
    p = sns.lineplot(data=data, x=x, y=y, hue=hue)
    p.set_title(title)
    p.figure.tight_layout()
    p.figure.savefig(out_path)
    p.cla()


def save_heatmap(last_epoch_df, metric, out_path):
    pivoted_df = last_epoch_df.pivot(index='user', columns='label', values='score')   
    cmap = 'crest' if metric == 'spearmanr' else 'crest_r'
    p = sns.heatmap(data=pivoted_df, annot=True, cmap=cmap, cbar=False)
    p.set_title(metric.upper())
    p.figure.tight_layout()
    p.figure.savefig(out_path)
    p.cla()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--models_input_directory', type=str, help='Models input directory.')
    args = parser.parse_args()

    out_plot_dir = 'results/eye_tracking/'

    if not os.path.exists(out_plot_dir):
        os.makedirs(out_plot_dir)

    loss_df = load_log_history_df(args.models_input_directory, parse_log_history_loss)
    eval_df = load_log_history_df(args.models_input_directory, parse_log_history_eval_metrics)
    
    save_lineplot(loss_df, 'epoch', 'loss', 'data', 'Training losses', os.path.join(out_plot_dir, 'training_losses.png'))
    
    for metric in ['mae', 'spearmanr']:
        last_epoch = int(eval_df['epoch'].max())
        save_lineplot(eval_df[eval_df['metric'] == metric], 'epoch', 'score', 'label', metric.upper(), 
                      os.path.join(out_plot_dir, f'{metric}_epochs.png'))
        last_epoch_df = eval_df[(eval_df['epoch'] == last_epoch) & (eval_df['metric'] == metric)]
        save_heatmap(last_epoch_df, metric, os.path.join(out_plot_dir, f'{metric}_epoch{last_epoch}.png')) 

if __name__ == '__main__':
    main()