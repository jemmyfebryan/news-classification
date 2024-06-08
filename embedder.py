import torch
import numpy as np

def get_sentence_embedding(tokenizer, bert_model, text: str, max_sequence_length: int):
    # _ensure_initialized()
    assert tokenizer is not None, "tokenizer not initialized"
    assert bert_model is not None, "bert_model not initialized"

    text_input = f"[CLS] {text.lower()} [SEP] [MASK]"

    tokenized_text = tokenizer.tokenize(text_input)
    segments = [1] * len(tokenized_text)

    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)

    tokens_tensor = torch.tensor([indexed_tokens])
    segments_tensors = torch.tensor([segments])

    with torch.no_grad():
        encoded_layers = bert_model(tokens_tensor, segments_tensors)

    last_hidden_state = encoded_layers.last_hidden_state

    token_vecs = last_hidden_state[0]

    sentence_embedding = [tensor for tensor in token_vecs]
    mask_token = sentence_embedding.pop()

    if len(sentence_embedding) > max_sequence_length:
        last_token = sentence_embedding.pop()
        sentence_embedding = sentence_embedding[:max_sequence_length-1]
        sentence_embedding.append(last_token)
    
    sentence_embedding = [mask_token]*(max_sequence_length-len(sentence_embedding)) + sentence_embedding

    return np.array(sentence_embedding)