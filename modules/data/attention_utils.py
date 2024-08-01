from transformers import AutoTokenizer
from torch.utils.data import Dataset
import json

def align_to_original_words(model_tokens: list, original_tokens: list, subword_prefix: str,
                            lowercase: bool = False) -> list:
    if lowercase:
        original_tokens = [tok.lower() for tok in original_tokens]
    model_tokens = model_tokens[1: -1]  # Remove <s> and </s>
    aligned_model_tokens = []
    alignment_ids = []
    alignment_id = -1
    orig_idx = 0
    for token in model_tokens:
        alignment_id += 1

        if token.startswith(subword_prefix):  # Remove the sub-word prefix
            token = token[len(subword_prefix):]
        if len(aligned_model_tokens) == 0:  # First token (serve?)
            aligned_model_tokens.append(token)
        elif aligned_model_tokens[-1] + token in original_tokens[orig_idx]:  # We are in the second (third, fourth, ...) sub-token
            aligned_model_tokens[-1] += token  # so we merge the token with its preceding(s)
            alignment_id -= 1
        elif aligned_model_tokens[-1] +' '+token in original_tokens[orig_idx]: # Sometimes there is a space in the original tokens
            aligned_model_tokens[-1] += ' '+token 
            alignment_id -= 1
        else:
            aligned_model_tokens.append(token)
        if aligned_model_tokens[-1] == original_tokens[orig_idx]:  # A token was equal to an entire original word or a set of
            orig_idx += 1  # sub-tokens was merged and matched an original word
        alignment_ids.append(alignment_id)

    if aligned_model_tokens != original_tokens:
        raise Exception(
            f'Failed to align tokens.\nOriginal tokens: {original_tokens}\nObtained alignment: {aligned_model_tokens}')
    return alignment_ids


def create_subwords_alignment(dataset: Dataset, tokenizer: AutoTokenizer, subword_prefix: str,
                              lowercase: bool = False) -> dict:
    sentence_alignment_dict = dict()

    for sent_id, sentence in zip(dataset['id'], dataset['text']):
        for tok_id, tok in enumerate(sentence):
            if tok == '–':
                sentence[tok_id] = '-'
            if 'ö' in tok:
                sentence[tok_id] = tok.replace('ö', '')
            if 'ô' in tok:
                sentence[tok_id] = tok.replace('ô', '')
            if 'û' in tok:
                sentence[tok_id] = tok.replace('û', '-')
        if lowercase:
            sentence = [word.lower() for word in sentence]
        tokenized_sentence = tokenizer(sentence, is_split_into_words=True, return_tensors='pt')
        input_ids = tokenized_sentence['input_ids'].tolist()[0]  # 0 because the batch_size is 1
        model_tokens = tokenizer.convert_ids_to_tokens(input_ids)
        alignment_ids = align_to_original_words(model_tokens, sentence, subword_prefix, lowercase)
        sentence_alignment_dict[sent_id] = {'model_input': tokenized_sentence, 'alignment_ids': alignment_ids}
    return sentence_alignment_dict


def save_dictionary(dictionary, out_path):
    with open(out_path, 'w') as out_file:
        out_file.write(json.dumps(dictionary))


def get_model_subword_prefix(tokenizer_name):
    if tokenizer_name == 'xlm-roberta-base' or tokenizer_name == 'idb-ita/gilberto-uncased-from-camembert':
        return '▁'
    elif tokenizer_name == 'roberta-base':
        return 'Ġ'
    else:
        return None