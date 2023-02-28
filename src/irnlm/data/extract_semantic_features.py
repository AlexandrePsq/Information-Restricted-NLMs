from irnlm.data.utils import get_function_words
from irnlm.data.text_tokenizer import tokenize


function_words = get_function_words()

def integral2semantic(path, language='english'):
    """Extract semantic features from the integral text.
    Args:
        - path: path
    Returns:
        - iterator: list of str (content words)
    """
    iterator = tokenize(path, language=language, with_punctuation=True, convert_numbers=True)
    iterator = [item.strip() for item in iterator]
    iterator = [' '.join([word for word in sent.split(' ') if word.lower() not in function_words]) for sent in iterator]
    iterator = [item for item in iterator if item !='']
    return iterator