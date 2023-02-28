import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import defaultdict

import benepar
from spacy.symbols import ORTH

import torch
from transformers import AutoTokenizer

from irnlm.data.utils import set_nlp_pipeline, get_ids_syntax
from irnlm.models.gpt2_syntactic.modeling_hacked_gpt2 import GPT2LMHeadModel
from irnlm.data.extract_syntactic_features import integral2syntactic



def load_model_and_tokenizer(trained_model='gpt2'):
    """Load a HuggingFace model and the associated tokenizer given its name.
    Args:
        - trained_model: str
    Returns:    
        - model: HuggingFace model
        - tokenizer: HuggingFace tokenizer
    """
    model = GPT2LMHeadModel.from_pretrained(trained_model)
    tokenizer = AutoTokenizer.from_pretrained(trained_model)
    return model, tokenizer

def pad_to_max_length(sequence, max_seq_length, space=1, special_token_end=2):
    """Pad sequence to reach max_seq_length
    Args:
        - sequence: list of int
        - max_seq_length: int
        - space: int (default 1)
        - special_token_end: int (default 2)
    Returns:
        - result: list of int
    """
    sequence = sequence[:max_seq_length]
    n = len(sequence)
    result = sequence + [space, special_token_end] * ((max_seq_length - n)// 2)
    if len(result)==max_seq_length:
        return result
    else:
        return result + [space]

def create_examples(sequence, max_seq_length, space=1, special_token_beg=0, special_token_end=2):
    """Returns list of InputExample objects.
    Args:
        - sequence: list of int
        - max_seq_length: int
        - space: int (default 1)
        - special_token_beg: int (default 0)
        - special_token_end: int (default 2)
    Returns:
        - result: list of int
    """
    return pad_to_max_length([special_token_beg] + sequence + [space, special_token_end], max_seq_length)

def batchify_pos_input(data, context_size=None, max_seq_length=512, special_token_beg=0, special_token_end=2, space=1):
    """Batchify sentence 'iterator' string, to get batches of sentences with a specific number of tokens per input.
    Function used with 'get_truncated_activations'.
    Arguments:
        - data: pos str
        - context_size: int
        - max_seq_length: int
    Returns:
        - input_ids: input batched
        - indexes: tuple of int
    """
    max_seq_length = max_seq_length if context_size is None else context_size+5 # +5 because of the special tokens + the current and following tokens
    
    if context_size==0:
        examples = [create_examples(
           data[i:i + 2], 
           max_seq_length, 
           special_token_beg=special_token_beg, 
           special_token_end=special_token_end, 
           space=space
           ) for i, _ in enumerate(data)]
    else:
        examples = [create_examples(
           data[i:i + context_size + 2], 
           max_seq_length, 
           special_token_beg=special_token_beg, 
           special_token_end=special_token_end, 
           space=space
           ) for i, _ in enumerate(data[:-context_size])]
    # the last example in examples has one element less from the input data, 
    # but it is compensated by the padding. we consider that the element 
    # following the last input token is the special token.
    features = [torch.FloatTensor(example).unsqueeze(0).to(torch.int64) for example in examples]
    input_ids = torch.cat(features, dim=0)
    indexes = [(1, context_size+2)] + [(context_size+1, context_size+2) for i in range(1, len(input_ids))] # shifted by one because of the initial special token
    # Cleaning
    del examples
    del features
    return input_ids, indexes

def extract_features(
    path, 
    model, 
    context_size=100, 
    max_seq_length=512, 
    bsz=32,
    FEATURE_COUNT=768,
    NUM_HIDDEN_LAYERS=12,
    ):
    """Extract the features from GPT-2.
    Args:
      - path: str
      - model: HuggingFace model
      - tokenizer: HuggingFace tokenizer
    """
    features = []
    benepar.download('benepar_en3')

    nlp = set_nlp_pipeline()
    special_case = [{ORTH: "hasnt"}]
    nlp.tokenizer.add_special_case("hasnt", special_case)
    benepar.download('benepar_en3')
    nlp.add_pipe('benepar', config={'model': 'benepar_en3'})
    print(nlp.pipe_names)

    transform_ids = get_ids_syntax()

    iterator = integral2syntactic(path, nlp, transform_ids, language='english')
    mapping = {i:[i] for i, j in enumerate(iterator)}
    print(f"Using context length of {context_size}.")
    

    input_ids, indexes = batchify_pos_input(
                                iterator, 
                                context_size=context_size, 
                                max_seq_length=max_seq_length, 
                                special_token_beg=0, 
                                special_token_end=2, 
                                space=1
                            )

    with torch.no_grad():
        hidden_states_activations_ = []
        for input_tmp in tqdm(input_ids.chunk(input_ids.size(0)//bsz)):
            hidden_states_activations_tmp = []
            encoded_layers = model(input_tmp, output_hidden_states=True)
            hidden_states_activations_tmp = np.stack([i.detach().numpy() for i in encoded_layers.hidden_states], axis=0) #shape: (#nb_layers, batch_size_tmp, max_seq_length, hidden_state_dimension)
            hidden_states_activations_.append(hidden_states_activations_tmp)

        hidden_states_activations_ = np.swapaxes(np.vstack([np.swapaxes(item, 0, 1) for item in hidden_states_activations_]), 0, 1) #shape: (#nb_layers, batch_size, max_seq_length, hidden_state_dimension)

    activations = []
    for i in range(hidden_states_activations_.shape[1]):
        index = indexes[i]
        activations.append([hidden_states_activations_[:, i, j, :] for j in range(index[0], index[1])])
    activations = np.stack([i for l in activations for i in l], axis=0)
    activations = np.swapaxes(activations, 0, 1) #shape: (#nb_layers, batch_size, hidden_state_dimension)

    for word_index in range(len(mapping.keys())):
        word_activation = []
        word_activation.append([activations[:, index, :] for index in mapping[word_index]])
        word_activation = np.vstack(word_activation)
        features.append(np.mean(word_activation, axis=0).reshape(-1))# list of elements of shape: (#nb_layers, hidden_state_dimension).reshape(-1)
    #After vstacking it will be of shape: (batch_size, #nb_layers*hidden_state_dimension)

    features = pd.DataFrame(np.vstack(features), columns=['hidden_state-layer-{}-{}'.format(layer, index) for layer in np.arange(1 + NUM_HIDDEN_LAYERS) for index in range(1, 1 + FEATURE_COUNT)])

    return features
