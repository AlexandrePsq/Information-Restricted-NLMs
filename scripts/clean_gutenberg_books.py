import argparse
from tqdm import tqdm
from joblib import Parallel, delayed  


from irnlm.data.utils import (
    get_punctuation, 
    get_word_list_and_freq,
    get_gutenberg_book_ids
    )
from irnlm.data.clean_dataset import clean_text

word_list, word_freq = get_word_list_and_freq()
punctuation = get_punctuation()
books = get_gutenberg_book_ids()
vocab = list(word_list) + punctuation


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Create dictionary')
    parser.add_argument("--path", type=int)
    parser.add_argument("--start", type=int)
    parser.add_argument("--stop", type=int)

    args = parser.parse_args()
    if args.path is not None:
        clean_text(args.path, vocab=vocab, filter_old_texts=True, save=True)
    else:
        # clean all gutenberg books in the interval `start:stop`
        start = int(args.start) if args.start is not None else 0
        stop = int(args.stop) if args.stop is not None else len(books)
        books = books[start:stop]
        
        Parallel(n_jobs=-1)(delayed(clean_text)(
            path, 
            vocab=vocab, 
            filter_old_texts=True, 
            save=True
        ) for path in tqdm(books))

