import os
import sys
sys.path.append(os.path.abspath("."))  
os.environ["CUDA_VISIBLE_DEVICES"] = "1" 

from modules.ValueZeroing.modeling.customized_modeling_roberta import RobertaForMaskedLMVZ
from modules.data.attention_utils import create_subwords_alignment, save_dictionary
from modules.data.dataset_utils import create_senteces_from_data
from transformers.tokenization_utils_base import BatchEncoding
from modules.modeling.model_utils import get_tokenizer_name
from sklearn.metrics.pairwise import cosine_distances
from transformers import AutoTokenizer, AutoConfig
from tqdm import tqdm
import pandas as pd
import numpy as np
import argparse
import torch
import re
import os

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def extract_sentence_valuezeroing_matrix(model: RobertaForMaskedLMVZ, tokenized_text:BatchEncoding):
    model.eval()
    with torch.no_grad():
        outputs = model(tokenized_text['input_ids'],
                        attention_mask=tokenized_text['attention_mask'],
                        output_hidden_states=True, output_attentions=False)
    org_hidden_states = torch.stack(outputs['hidden_states']).squeeze(1)
    input_shape = tokenized_text['input_ids'].size()
    _, seq_length = input_shape
    
    score_matrix = np.zeros((model.config.num_hidden_layers, seq_length, seq_length))
    layers_modules = model.roberta.encoder.layer
        
    for l, layer_module in enumerate(layers_modules):
        for t in range(seq_length):
            extended_blanking_attention_mask: torch.Tensor = model.roberta.get_extended_attention_mask(
                        tokenized_text['attention_mask'], input_shape)
            with torch.no_grad():
                layer_outputs = layer_module(org_hidden_states[l].unsqueeze(0),  # previous layer's original output
                                                    attention_mask=extended_blanking_attention_mask,
                                                    output_attentions=False,
                                                    zero_value_index=t,
                                                    )

            hidden_states = layer_outputs[0].squeeze().detach().cpu().numpy()
            x = hidden_states
            y = org_hidden_states[l + 1].detach().cpu().numpy()
            distances = cosine_distances(x, y).diagonal()

            score_matrix[l, :, t] = distances
            
    valuezeroing_scores = score_matrix / np.sum(score_matrix, axis=-1, keepdims=True)
    return valuezeroing_scores

def aggregate_tokens_attention(sentence_attention:list, alignment_ids:list):
    sentence_attention = sentence_attention[1:-1] # remove <s> and </s>
    aggregated_weights = [[] for _ in range(len(set(alignment_ids)))]
    for al_id, att_weights in zip(alignment_ids, sentence_attention):
        aggregated_weights[al_id].append(att_weights)

    # si prende solo il primo token
    aggregated_weights = [el[0] for el in aggregated_weights]
    return aggregated_weights

def extract_valuezeroing_scores(model: RobertaForMaskedLMVZ, tokenized_text:BatchEncoding, alignment_ids:list, sentence_aggregation_method:str='cls'):
    valuezeroing_matrix = extract_sentence_valuezeroing_matrix(model, tokenized_text)
    layers_scores = []
    for layer in range(valuezeroing_matrix.shape[0]):
        layer_contributions = valuezeroing_matrix[layer]
        if sentence_aggregation_method == 'avg':
            sentence_scores = np.mean(layer_contributions, axis=0).tolist()
        elif sentence_aggregation_method == 'cls':
            sentence_scores = layer_contributions[0].tolist()
        else:
            raise Exception(f'Method {sentence_aggregation_method} not implemented')
        sentence_scores = aggregate_tokens_attention(sentence_scores, alignment_ids)
        layers_scores.append(sentence_scores)
    return layers_scores

def get_valuezeroing_scores(model, subwords_alignment_dict):
    valuezeroing_scores = {layer: dict() for layer in range(12)}
    for sent_id in tqdm(subwords_alignment_dict):
        tokenized_text = subwords_alignment_dict[sent_id]['model_input'].to(device)
        alignment_ids = subwords_alignment_dict[sent_id]['alignment_ids']
        layers_scores = extract_valuezeroing_scores(model, tokenized_text, alignment_ids)
        for layer in range(len(layers_scores)):
            valuezeroing_scores[layer][sent_id] = layers_scores[layer]
    return valuezeroing_scores


def get_subword_prefix(tokenizer_name):
    if 'roberta' in tokenizer_name.lower():
        return'Ġ'
    else:
        raise Exception(f'Model {tokenizer_name} not supported yet.')


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

    config = AutoConfig.from_pretrained(args.model_path) 
    model = RobertaForMaskedLMVZ.from_pretrained(args.model_path, config=config)       
    model.to(device)

    tokenizer_name = get_tokenizer_name(args.model_path)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, add_prefix_space=True)
    subword_prefix = get_subword_prefix(tokenizer_name)

    sentence_alignment_dict = create_subwords_alignment(test_dataset, tokenizer, subword_prefix)
    valuezeroing_scores = get_valuezeroing_scores(model, sentence_alignment_dict)

    if not os.path.exists(args.output_directory):
        os.makedirs(args.output_directory)

    for layer in range(12):
        out_path = os.path.join(args.output_directory, f'{layer}.json')
        layer_contribs = valuezeroing_scores[layer]
        save_dictionary(layer_contribs, out_path)


if __name__ == '__main__':
    main()