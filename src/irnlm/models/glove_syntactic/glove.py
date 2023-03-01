import os
import gdown
import numpy as np
import pandas as pd
from tqdm import tqdm

import benepar
from spacy.symbols import ORTH

from irnlm.data.utils import set_nlp_pipeline, get_ids_syntax
from irnlm.data.extract_syntactic_features import integral2syntactic



def load_model_and_tokenizer(trained_model='../data/glove.6B.300d.txt'):
    """Load a GloVe model given its name.
    Download Glove weights from URL if not already done.
    Args:
        - trained_model: str
    Returns:
        - model: dict
        - tokenizer: None
    """
    if not os.path.exists(trained_model):
        url = "https://nlp.stanford.edu/data/glove.6B.zip"
        output = './data/glove.zip'
        gdown.download(url, output, quiet=False)
        os.system(f'unzip {output}')
    model = init_embeddings(trained_model=trained_model)
    return model, None


def init_embeddings(trained_model='../data/glove.6B.300d.txt'):
    """ Initialize an instance of GloVe Dictionary.
    Args:
        - trained_model: str
    Returns:
        - model: dict
    """
    model = {}
    with open(trained_model, 'r', encoding="utf-8") as f: 
        for line in f: 
            values = line.split() 
            word = values[0] 
            vector = np.asarray(values[1:], "float32") 
            model[word] = vector 
    return model

def update_model(glove_model, embedding_size=300):
    """ Ad some words to a glove model.
    Args:
        - glove_model: dict
    Returns:
        - glove_model: dict
    """
    words2add = {} # the second value in the tuple is the number of following words to skip in generate
    for key in words2add.keys():
        if key not in glove_model.keys():
            glove_model[key] = np.zeros((embedding_size,))
            for word in words2add[key][0]:
                try:
                    glove_model[key] += glove_model[word]
                except:
                    print(f'{word} does not appear in the vocabulary... Be sure that it is normal.')
            glove_model[key] = glove_model[key] / len(words2add[key][0])
    return glove_model

def extract_features(
    path, 
    model, 
    FEATURE_COUNT=300,
    ):
    """Extract the features from GloVe.
    Args:
        - path: list of str
        - model: GloVe model
    Returns:
        - features: csv
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

    words = integral2syntactic(path, nlp, transform_ids, language='english') # not words but ids

    columns = ['embedding-{}'.format(i) for i in range(1, 1 + FEATURE_COUNT)]
    features = []
    for item in tqdm(words):
        if item not in model.keys():
            item = '<unk>'
        features.append(model[item])

    features = pd.DataFrame(np.vstack(features), columns=columns)
    return features