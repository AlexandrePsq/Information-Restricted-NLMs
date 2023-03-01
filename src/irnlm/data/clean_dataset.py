import os, glob, re
import argparse
from tqdm import tqdm
from joblib import Parallel, delayed  
import gutenberg_cleaner


from irnlm.data.utils import (
    get_punctuation, 
    get_word_list_and_freq,
    get_gutenberg_book_ids
    )
from irnlm.data.text_tokenizer import tokenize

word_list, word_freq = get_word_list_and_freq()
punctuation = get_punctuation()
books = get_gutenberg_book_ids()


def clean_text(path, filter_old_texts=True, save=True, vocab=None):
    """Clean input text file.
    Remove books that are too ancient.
    Filter with a given vocabulary.
    """
    try: 
        text = open(path, 'r').read()
    except: 
        print(path, '--> Error') 
        text = open(path, 'r', encoding="utf8", errors='ignore').read() 
        text = text.encode("utf-8")
        with open(path, 'wb') as fout: fout.write(text)
        text = open(path, 'r').read() 

    # Filter old texts
    matches = re.findall('\\n\\n(?:19|20)\d{2}', text[:2000]) # check if there is publication date among 2000 first characters
    if (len(matches) > 0) or filter_old_texts:
        # Replace numbers (optional) 
        #transf = inflect.engine()
        #numbers = re.findall('\d+', text)
        #for number in numbers:
        #    text = text.replace(number, transf.number_to_words(number))

        # Separating puntcuation from words witha space

        text = gutenberg_cleaner.super_cleaner(text)
        text = text.replace('Mr.', 'Mr').replace('mr.', 'mr').replace('Mrs.', 'Mrs').replace('mrs.', 'mrs')
        text = re.sub(' +', ' ', text)
        text = re.sub('\.{3}\.+', '\.{3}', text)
        text = re.sub('(\\n)+', '\\n', text)
        text = re.sub('\[[^\[]*\]', '', text)
        text = text.replace('\n', ' ').replace("\'", " ' ").replace('_', ' ').replace('---', ' ').replace('--', ' - ').replace('. . .', '...')
        text = tokenize(text, 'english', with_punctuation=True, convert_numbers=True)
        text = [sent.lower() for sent in text]
        text = filter_with_vocab(text, vocab=vocab)  

    directory = os.path.dirname(path)
    name = os.path.basename(path)[:-4]
    
    if save:
        with open(os.path.join(directory, name + '_cleaned.txt'), 'w') as f:
            text = ' '.join(text)
            f.write(text)
    else:
        return text

def filter_with_vocab(sentences: list, vocab: list) -> bool:
    """If one word of a sentence is not in vocabulary, it is removed.
    Returns paragraph with non-valid sentences removed.
    """
    if vocab is None:
        return sentences

    def is_valid(sentence):
        words = sentence.split()
        result = True
        word_index = 0
        limit = len(words)
        while (word_index < limit) and result:
            result = words[word_index] in vocab
            word_index += 1
        return result

    # It is better to parallelize over input texts than sentences
    result = []
    for sentence in sentences:
        if is_valid(sentence):
            result.append(sentence)
    #are_valid = Parallel(n_jobs=-1)(delayed(is_valid)(sentence) for sentence in sentences)
    #result = [sentences[index] for index, is_valid in enumerate(are_valid) if is_valid]
    
    return result