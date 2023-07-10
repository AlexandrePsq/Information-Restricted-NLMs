import os
import re
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch

from irnlm.data.extract_semantic_features import integral2semantic
from irnlm.models.gpt2.extract_features_gpt2_integral import batchify_to_truncated_input


def extract_features(
    path,
    model,
    nlp_tokenizer,
    context_size=100,
    max_seq_length=512,
    bsz=32,
    language="english",
    FEATURE_COUNT=768,
    NUM_HIDDEN_LAYERS=12,
    n_jobs=5,
    convert_numbers=False,
):
    """Extract the features from GPT-2.
    Args:
        - path: str
        - model: HuggingFace model
        - nlp_tokenizer: HuggingFace tokenizer
    """
    features = []
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    iterator = integral2semantic(
        path, language=language, n_jobs=n_jobs, convert_numbers=convert_numbers
    )
    iterator = " ".join(iterator)

    # Computing mapping
    items = [nlp_tokenizer.tokenize(i) for i in iterator.split(" ")]
    count = 0
    mapping = {}
    for i, j in enumerate(items):
        for k in j:
            if i not in mapping.keys():
                mapping[i] = [count]
            else:
                mapping[i] += [count]
            count += 1
    # ids = nlp_tokenizer(iterator).word_ids()
    # unique_ids = np.unique(ids)
    # mapping = {
    #    i: list(np.where(ids == i)[0]) for i in unique_ids
    # }  # match_tokenized_to_untokenized(tokenized_text, iterator)

    input_ids, indexes, tokens = batchify_to_truncated_input(
        iterator,
        nlp_tokenizer,
        context_size=context_size,
        max_seq_length=max_seq_length,
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
    )  # shape: (#nb_layers, #nb_tokens, hidden_state_dimension)

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
