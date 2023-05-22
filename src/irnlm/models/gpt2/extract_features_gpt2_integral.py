import os
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
from transformers import AutoTokenizer

from irnlm.models.tokenizer import read_bpe
from irnlm.data.text_tokenizer import tokenize
from irnlm.models.gpt2.modeling_hacked_gpt2_integral import GPT2LMHeadModel
from irnlm.models.gpt2.modeling_hacked_gpt2_semantic import (
    GPT2LMHeadModel as GPT2LMHeadModelSemantic,
)


def load_model_and_tokenizer(
    trained_model="gpt2", model_type="integral", max_length=512
):
    """Load a HuggingFace model and the associated tokenizer given its name.
    Args:
        - trained_model: str
    Returns:
        - model: HuggingFace model
        - tokenizer: HuggingFace tokenizer
    """
    tokenizer_path = os.path.join(trained_model, "bpe-vocab.json")
    if not os.path.exists(tokenizer_path):
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    else:
        tokenizer = read_bpe(
            path=tokenizer_path,
            max_length=max_length,  # max_length
        )
    try:
        special_token_pad_ids = tokenizer(tokenizer.pad_token)["input_ids"][0]
    except ValueError:
        special_token_pad_ids = None  # 50256

    if model_type in ["integral", "syntactic"]:
        model = GPT2LMHeadModel.from_pretrained(
            trained_model,
            output_hidden_states=True,
            output_attentions=False,
            pad_token_id=special_token_pad_ids,
        )
    elif model_type == "semantic":
        model = GPT2LMHeadModelSemantic.from_pretrained(
            trained_model,
            output_hidden_states=True,
            output_attentions=False,
            pad_token_id=special_token_pad_ids,
        )
    return model, tokenizer


def pad_to_max_length(
    sequence, max_seq_length, space=220, special_token_end=50256, special_token_pad=None
):
    """Pad sequence to reach max_seq_length
    Args:
        - sequence: list of int
        - max_seq_length: int
        - space: int (default 220)
        - special_token_end: int (default 50256)
    Returns:
        - result: list of int
    """
    sequence = sequence[:max_seq_length]
    n = len(sequence)
    if special_token_pad is None:
        result = sequence + [space, special_token_end] * ((max_seq_length - n) // 2)
        if len(result) == max_seq_length:
            return result
        else:
            return result + [space]
    else:
        result = sequence + [special_token_pad] * (max_seq_length - n)
        if len(result) == max_seq_length:
            return result
        else:
            return result + [special_token_pad]


def create_examples(
    sequence,
    max_seq_length,
    space=220,
    special_token_beg=50256,
    special_token_end=50256,
    special_token_pad=None,
):
    """Returns list of InputExample objects.
    Args:
        - sequence: list of int
        - max_seq_length: int
        - space: int (default 220)
        - special_token_beg: int (default 50256)
        - special_token_end: int (default 50256)
    Returns:
        - result: list of int
    """
    if space is not None:
        return pad_to_max_length(
            [special_token_beg] + sequence + [space, special_token_end],
            max_seq_length,
            space=space,
            special_token_pad=special_token_pad,
        )
    else:
        return pad_to_max_length(
            [special_token_beg] + sequence + [special_token_end],
            max_seq_length,
            space=space,
            special_token_pad=special_token_pad,
        )


def batchify_to_truncated_input(
    iterator,
    tokenizer,
    context_size=None,
    max_seq_length=512,
):
    """Batchify sentence 'iterator' string, to get batches of sentences with a specific number of tokens per input.
    Function used with 'get_truncated_activations'.
    Arguments:
        - iterator: sentence str
        - tokenizer: Tokenizer object
        - context_size: int
        - max_seq_length: int
        - space: str (default 'Ġ')
    Returns:
        - input_ids: input batched
        - indexes: tuple of int
    """
    special_token_beg = tokenizer.bos_token
    special_token_end = tokenizer.eos_token
    special_token_beg_ids = tokenizer(tokenizer.bos_token)["input_ids"][0]
    special_token_end_ids = tokenizer(tokenizer.eos_token)["input_ids"][0]
    try:
        special_token_pad = tokenizer.pad_token
        special_token_pad_ids = tokenizer(tokenizer.pad_token)["input_ids"][0]
        space = None
    except ValueError:
        special_token_pad = None
        special_token_pad_ids = None
        space = 220

    max_seq_length = (
        max_seq_length if context_size is None else context_size + 5
    )  # +5 because of the special tokens + the current and following tokens
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    try:
        data = tokenizer.encode(iterator).ids
        text = tokenizer.encode(iterator).tokens
    except:
        data = tokenizer.encode(iterator)
        text = tokenizer.tokenize(iterator)

    if context_size == 0:
        examples = [
            create_examples(
                data[i : i + 2],
                max_seq_length,
                space=space,
                special_token_beg=special_token_beg_ids,
                special_token_end=special_token_end_ids,
                special_token_pad=special_token_pad_ids,
            )
            for i, _ in enumerate(data)
        ]
        tokens = [
            create_examples(
                text[i : i + 2],
                max_seq_length,
                space=space,
                special_token_beg=special_token_beg,
                special_token_end=special_token_end,
                special_token_pad=special_token_pad,
            )
            for i, _ in enumerate(text)
        ]
    else:
        examples = [
            create_examples(
                data[i : i + context_size + 2],
                max_seq_length,
                space=space,
                special_token_beg=special_token_beg_ids,
                special_token_end=special_token_end_ids,
                special_token_pad=special_token_pad_ids,
            )
            for i, _ in enumerate(data[:-context_size])
        ]
        tokens = [
            create_examples(
                text[i : i + context_size + 2],
                max_seq_length,
                space=space,
                special_token_beg=special_token_beg,
                special_token_end=special_token_end,
                special_token_pad=special_token_pad,
            )
            for i, _ in enumerate(text[:-context_size])
        ]
    # the last example in examples has one element less from the input data, but it is compensated by the padding. we consider that the element following the last input token is the special token.
    features = [
        torch.FloatTensor(example).unsqueeze(0).to(torch.int64) for example in examples
    ]
    input_ids = torch.cat(features, dim=0)
    indexes = [(1, context_size + 2)] + [
        (context_size + 1, context_size + 2) for i in range(1, len(input_ids))
    ]  # shifted by one because of the initial special token
    # Cleaning
    del examples
    del features
    return input_ids, indexes, tokens


def extract_features(
    path,
    model,
    tokenizer,
    context_size=100,
    max_seq_length=512,
    space="Ġ",
    bsz=32,
    language="english",
    special_token_beg="<|endoftext|>",
    special_token_end="<|endoftext|>",
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
    iterator = tokenize(
        path, language=language, with_punctuation=True, convert_numbers=True
    )
    iterator = [item.strip() for item in iterator]

    ids = tokenizer(iterator).word_ids()
    unique_ids = np.unique(ids)
    mapping = {
        i: list(np.where(ids == i)[0]) for i in unique_ids
    }  # match_tokenized_to_untokenized(tokenized_text, iterator)

    input_ids, indexes, tokens = batchify_to_truncated_input(
        iterator,
        tokenizer,
        context_size=context_size,
        max_seq_length=max_seq_length,
        space=space,
        special_token_beg=special_token_beg,
        special_token_end=special_token_end,
    )

    with torch.no_grad():
        hidden_states_activations_ = []
        for input_tmp in tqdm(input_ids.chunk(input_ids.size(0) // bsz)):
            hidden_states_activations_tmp = []
            encoded_layers = model(input_tmp, output_hidden_states=True)
            hidden_states_activations_tmp = np.stack(
                [i.detach().numpy() for i in encoded_layers.hidden_states], axis=0
            )  # shape: (#nb_layers, batch_size_tmp, max_seq_length, hidden_state_dimension)
            hidden_states_activations_.append(hidden_states_activations_tmp)

        hidden_states_activations_ = np.swapaxes(
            np.vstack([np.swapaxes(item, 0, 1) for item in hidden_states_activations_]),
            0,
            1,
        )  # shape: (#nb_layers, batch_size, max_seq_length, hidden_state_dimension)

    activations = []
    for i in range(hidden_states_activations_.shape[1]):
        index = indexes[i]
        activations.append(
            [hidden_states_activations_[:, i, j, :] for j in range(index[0], index[1])]
        )
    activations = np.stack([i for l in activations for i in l], axis=0)
    activations = np.swapaxes(
        activations, 0, 1
    )  # shape: (#nb_layers, batch_size, hidden_state_dimension)

    for word_index in range(len(mapping.keys())):
        word_activation = []
        word_activation.append(
            [activations[:, index, :] for index in mapping[word_index]]
        )
        word_activation = np.vstack(word_activation)
        features.append(
            np.mean(word_activation, axis=0).reshape(-1)
        )  # list of elements of shape: (#nb_layers, hidden_state_dimension).reshape(-1)
    # After vstacking it will be of shape: (batch_size, #nb_layers*hidden_state_dimension)

    features = pd.DataFrame(
        np.vstack(features),
        columns=[
            "hidden_state-layer-{}-{}".format(layer, index)
            for layer in np.arange(1 + NUM_HIDDEN_LAYERS)
            for index in range(1, 1 + FEATURE_COUNT)
        ],
    )

    return features
