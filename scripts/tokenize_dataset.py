import os
import argparse

from transformers import GPT2Tokenizer  

from irnlm.utils import save_pickle

os.environ["TOKENIZERS_PARALLELISM"] = "true"

if __name__=='__main__':
    
    parser = argparse.ArgumentParser(description="""Tokenize a Dataset using the right tokenizer (default, semantic, syntactic).""")
    parser.add_argument("--path", type=str)
    parser.add_argument("--type", type=str, default='default') #'semantic', 'syntactic'

    args = parser.parse_args()
    if args.nsplits=='syntactic':
        print('The Syntactic Dataset is tokenized by default.')
    else:
        # Reading data
        print(f"Loading {args.path} ...")
        dataset = open(args.path, 'r').read()
        n = len(dataset)
        print(f'The Input text contains {n} elements.')

        # Creating name
        saving_folder = os.path.dirname(args.path)
        name = os.path.basename(args.path).split('.')[0]
        if args.nsplits=='semantic':
            name += '_semantic'

        # Loading tokenizer
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2') # you can customize it if you want
        dataset_ids = tokenizer.encode(dataset).ids # you will have a warning, but it is not important

        save_pickle(os.path.join(saving_folder, name+'.pkl'), dataset_ids)

