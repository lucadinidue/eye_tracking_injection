import os
import sys
sys.path.append(os.path.abspath(".")) 

import torch
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from modules.data.attention_utils import create_subwords_alignment, save_dictionary
from modules.data.dataset_utils import create_senteces_from_data
from modules.modeling.model_utils import get_tokenizer_name
from transformers.tokenization_utils_base import BatchEncoding
from transformers import AutoTokenizer, AutoModelForMaskedLM
from tqdm import tqdm
import pandas as pd
import argparse
import re

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    
    
def get_subword_prefix(tokenizer_name):
    if 'roberta' in tokenizer_name.lower():
        return'Ġ'
    else:
        raise Exception(f'Model {tokenizer_name} not supported yet.')
    
def aggregate_tokens_attention(sentence_attention:list, alignment_ids:list):
    sentence_attention = sentence_attention[1:-1] # remove <s> and </s>
    aggregated_weights = [[] for _ in range(len(set(alignment_ids)))]
    for al_id, att_weights in zip(alignment_ids, sentence_attention):
        aggregated_weights[al_id].append(att_weights)

    # si prende solo il primo token
    aggregated_weights = [el[0] for el in aggregated_weights]
    return aggregated_weights

def extract_sentence_attention(model: AutoModelForMaskedLM, tokenized_text:BatchEncoding, alignment_ids:list, sentence_aggregation_method:str='cls') -> list:
    model_output = model(**tokenized_text)
    attention_matrices = model_output['attentions']
    layers_attentions = []
    for layer in range(len(attention_matrices)):
        layer_attention_matrix = attention_matrices[layer].detach().squeeze()
        avg_attention_matrix = torch.mean(layer_attention_matrix, dim=0) # media tra le teste di attenzione
        if sentence_aggregation_method == 'avg':
            sentence_attention = torch.mean(avg_attention_matrix, dim=0).tolist()
        elif sentence_aggregation_method == 'cls':
            sentence_attention = avg_attention_matrix[:,0].tolist()
        else:
            raise Exception(f'Method {sentence_aggregation_method} not implemented')
        sentence_attention = aggregate_tokens_attention(sentence_attention, alignment_ids)
        layers_attentions.append(sentence_attention)
    return layers_attentions


def get_attention_weights(model:AutoModelForMaskedLM, subwords_alignment_dict):
    attention_weights = {layer:{} for layer in range(12)}
    for sent_id in tqdm(subwords_alignment_dict):
        tokenized_text = subwords_alignment_dict[sent_id]['model_input'].to(device)
        alignment_ids = subwords_alignment_dict[sent_id]['alignment_ids']
        layers_attentions = extract_sentence_attention(model, tokenized_text, alignment_ids)
        for layer in range(len(layers_attentions)):
            attention_weights[layer][sent_id] = layers_attentions[layer]
    return attention_weights


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model_path', dest='model_path', type=str)
    parser.add_argument('-o', '--output_directory', dest='output_directory', type=str)
    args = parser.parse_args()

    try:
        user_id = re.findall(r'pp(\d*)', args.model_path)[0]
    except: # the not finetuned model has no user
        user_id = 21
    test_path = f'data/geco/dataset/pp{user_id}_dataset_test.csv'

    test_df = pd.read_csv(test_path, index_col=0)
    test_dataset = create_senteces_from_data(test_df, [], keep_id=True)

    if not os.path.exists(args.output_directory):
        os.makedirs(args.output_directory)

    model = AutoModelForMaskedLM.from_pretrained(args.model_path, output_attentions=True).to(device)
    tokenizer_name = get_tokenizer_name(args.model_path)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, add_prefix_space=True)
    subword_prefix = get_subword_prefix(tokenizer_name)

    sentence_alignment_dict = create_subwords_alignment(test_dataset, tokenizer, subword_prefix)
    attention_weights = get_attention_weights(model, sentence_alignment_dict)


    for layer in range(12):
        output_path = os.path.join(args.output_directory, f'{layer}.json')
        save_dictionary(attention_weights[layer], output_path)





if __name__ == '__main__':
    main()