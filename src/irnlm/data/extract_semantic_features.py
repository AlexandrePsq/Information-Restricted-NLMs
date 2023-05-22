from tqdm import tqdm
from joblib import Parallel, delayed

from irnlm.data.utils import (
    get_negations,
    get_pronouns,
    get_positions_words,
    get_quantity_words,
    get_function_words,
    get_punctuation,
)
from irnlm.data.text_tokenizer import tokenize


def get_mapping(language):
    """ """
    mapping = {word: "PRON" for word in get_pronouns(language)}
    mapping.update({word: "POSITION" for word in get_positions_words(language)})
    mapping.update({word: "NEG" for word in get_negations(language)})
    mapping.update({word: "QUANTITY" for word in get_quantity_words(language)})
    return mapping


def clean_sentence(sent, mapping, function_words):
    """Replace pronouns with PRON.
    Replace positions with POSITION.
    Replace negations with NEG.
    Replace quantity with QUANTITY.
    Remove function words.
    """
    words = sent.split(" ")
    words = [
        mapping[word.lower()] if word.lower() in mapping.keys() else word
        for word in words
    ]
    sent = " ".join([word for word in words if word.lower() not in function_words])
    return sent


def integral2semantic(path, language="english", n_jobs=-1, convert_numbers=False):
    """Extract semantic features from the integral text.
    Args:
        - path: path
        - n_jobs: int
        - convert_numbers: bool
    Returns:
        - iterator: list of str (content words)
    """
    mapping = get_mapping(language)
    function_words = get_function_words(language=language) + get_punctuation()
    iterator = tokenize(
        path, language=language, with_punctuation=True, convert_numbers=convert_numbers
    )
    iterator = [item.strip() for item in iterator]
    n = len(iterator)
    iterator = Parallel(n_jobs=n_jobs, batch_size=min(1000, n))(
        delayed(clean_sentence)(sent, mapping, function_words)
        for sent in tqdm(iterator, desc="Cleaning sentences", total=n)
    )
    # iterator = [clean_sentence(sent, mapping, function_words) for sent in tqdm(iterator, desc='Cleaning sentences', total=len(iterator))]
    iterator = [item for item in iterator if item != ""]
    return iterator
