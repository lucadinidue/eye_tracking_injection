def get_tokenizer_name(model_name:str) -> str:
    if 'roberta' in model_name.lower():
        return 'FacebookAI/roberta-base'
    else:
        raise Exception(f'Model {model_name} not supported yet.')