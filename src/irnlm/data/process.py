# basis imports
import os
import re
import glob
import time
import numpy as np
import argparse
import pandas as pd
from tqdm import tqdm
from itertools import combinations
try:
    import pickle5 as pickle
except:
    import pickle
import utils

# Utilities
from joblib import Parallel, delayed

syntax = __import__('04_syntax_generator')
tokenizer = __import__('05_tokenizer')


import spacy
import benepar
from spacy.symbols import ORTH
from benepar import BeneparComponent


def save_object(filename, data):
    """Save computed examples and features.
    """
    with open(filename, 'wb') as outp:  # Overwrites any existing file.
        pickle.dump(data, outp, pickle.HIGHEST_PROTOCOL)

def load_object(filename):
    """Load computed examples and features.
    """
    with open(filename, 'rb') as inp:  # Overwrites any existing file.
        data = pickle.load(inp)
    return data

def create_examples( sequence):
    """Returns list of InputExample objects."""
    return pad_to_max_length([0] + sequence + [225, 2])

def pad_to_max_length(sequence, max_seq_length=512):
    """Pad sequence to reach max_seq_length"""
    sequence = sequence[:max_seq_length]
    n = len(sequence)
    result = sequence + [225, 1] * ((max_seq_length - n)// 2)
    if len(result)==max_seq_length:
        return result
    else:
        return result + [225]

def extract_syntax(doc): 
    """Extract number of closing nodes for each words of an input sequence. 
    """ 
    ncn = [] 
    morph = []
    pos_ = []
    for sent in doc.sents: 
        parsed_string = sent._.parse_string 
        words = sent.text.split(' ') 
        for token in sent: 
            m = str(morphs.index(str(token.morph))) if str(token.morph)!='' else '0'
            if len(m)==1:
                m = '0' + m
            morph.append(m)
            p = str(pos.index(token.pos_))
            if len(p)==1:
                p = '0' + p
            pos_.append(p)
            word = token.text+')' 
            index = parsed_string.find(word)
            l = len(word) 
            count = 1 
            i = index+l 
            while i<len(parsed_string) and parsed_string[i]==')' :
                count+=1
                i+=1
            ncn.append(str(min(count, 9))) # we take into account a maximum of 9 closing parenthesis
            parsed_string = parsed_string[i:] 
    return [int(''.join(items)) for items in list(zip(ncn, morph, pos_))]

def text_to_doc(nlp, text, n_split_internal=500, name='_train_', index=0):
    """Process text into sytactic representaitons...
    """
    result = []
    n = len(text)
    print('Text length: ', n)
    for i in tqdm(range(n_split_internal)):
        data = text[i*n//n_split_internal: (i+1)*n//n_split_internal]
        try:
            data = ' . '.join(data.split('.')[:-1])+ ' .'
        except:
            print('failed to parse sentences...')
            data = '.'.join([str(k) for k in data.split('.')][:-1])+ ' .'
        print('Tokenizing...')
        data_tmp = tokenizer.tokenize(data, language='english', train=False, with_punctuation=True, convert_numbers=True)
        print('Parsing...')
        tmp = []
        #doc = nlp(' '.join(data_tmp))
        for sent in data_tmp:
            try:
                doc = nlp(sent)
                tmp.append(extract_syntax(doc))
            except:
                utils.write(os.path.join(data_folder, 'tmp', f'logs_{index}.txt'), sent+'\n\n\n')
        result.append([i for l in tmp for i in l])
        #print('Retrieving syntax...')
        #result.append(extract_syntax(doc))
    result = [i for l in result for i in l]
    save_object(os.path.join(data_folder, 'tmp', 'gpt2' + name + f'syntax_{index}.pkl'), result)


benepar.download('benepar_en3')

nlp = spacy.load("en_core_web_lg")
nlp.remove_pipe("ner")
nlp.max_length = np.inf
special_case = [{ORTH: "hasnt"}]
nlp.tokenizer.add_special_case("hasnt", special_case)

nlp.add_pipe('benepar', config={'model': 'benepar_en3'})
#nlp.add_pipe(benepar.BeneparComponent("benepar_en3"))

print(nlp.pipe_names)


data_folder = '/neurospin/unicog/protocols/IRMf/LePetitPrince_Pallier_2018/LePetitPrince/data/text/english/all_training'
files = [os.path.join(data_folder, item) for item in ['gpt2_train.txt', 'gpt2_test.txt', 'gpt2_dev.txt']]

morphs = ['AdvType=Ex', 'Aspect=Perf|Tense=Past|VerbForm=Part', 'Aspect=Prog|Tense=Pres|VerbForm=Part', 'Case=Acc|Gender=Fem|Number=Sing|Person=3|PronType=Prs', 'Case=Acc|Gender=Fem|Number=Sing|Person=3|PronType=Prs|Reflex=Yes', 'Case=Acc|Gender=Masc|Number=Sing|Person=3|PronType=Prs', 'Case=Acc|Gender=Masc|Number=Sing|Person=3|PronType=Prs|Reflex=Yes', 'Case=Acc|Gender=Neut|Number=Sing|Person=3|PronType=Prs|Reflex=Yes', 'Case=Acc|Number=Plur|Person=1|PronType=Prs', 'Case=Acc|Number=Plur|Person=1|PronType=Prs|Reflex=Yes', 'Case=Acc|Number=Plur|Person=3|PronType=Prs', 'Case=Acc|Number=Plur|Person=3|PronType=Prs|Reflex=Yes', 'Case=Acc|Number=Sing|Person=1|PronType=Prs', 'Case=Acc|Number=Sing|Person=1|PronType=Prs|Reflex=Yes', 'Case=Acc|Person=2|PronType=Prs|Reflex=Yes', 'Case=Nom|Gender=Fem|Number=Sing|Person=3|PronType=Prs', 'Case=Nom|Gender=Masc|Number=Sing|Person=3|PronType=Prs', 'Case=Nom|Number=Plur|Person=1|PronType=Prs', 'Case=Nom|Number=Plur|Person=3|PronType=Prs', 'Case=Nom|Number=Sing|Person=1|PronType=Prs', 'ConjType=Cmp', 'Definite=Def|PronType=Art', 'Definite=Ind|PronType=Art', 'Degree=Cmp', 'Degree=Pos', 'Degree=Sup', 'Foreign=Yes', 'Gender=Fem|Number=Sing|Person=3|Poss=Yes|PronType=Prs', 'Gender=Fem|Number=Sing|Person=3|Poss=Yes|PronType=Prs|Reflex=Yes', 'Gender=Masc|Number=Sing|Person=3|Poss=Yes|PronType=Prs', 'Gender=Masc|Number=Sing|Person=3|Poss=Yes|PronType=Prs|Reflex=Yes', 'Gender=Neut|Number=Sing|Person=3|Poss=Yes|PronType=Prs', 'Gender=Neut|Number=Sing|Person=3|PronType=Prs', 'Hyph=Yes', 'Mood=Ind|Number=Sing|Person=3|Tense=Past|VerbForm=Fin', 'Mood=Ind|Number=Sing|Person=3|Tense=Pres|VerbForm=Fin', 'Mood=Ind|Person=1|Tense=Pres|VerbForm=Fin', 'Mood=Ind|Tense=Pres|VerbForm=Fin', 'None', 'NounType=Prop|Number=Plur', 'NounType=Prop|Number=Sing', 'NumType=Card', 'NumType=Ord', 'Number=Plur', 'Number=Plur|Person=1|Poss=Yes|PronType=Prs', 'Number=Plur|Person=1|Poss=Yes|PronType=Prs|Reflex=Yes', 'Number=Plur|Person=2|Poss=Yes|PronType=Prs|Reflex=Yes', 'Number=Plur|Person=3|Poss=Yes|PronType=Prs', 'Number=Plur|Person=3|Poss=Yes|PronType=Prs|Reflex=Yes', 'Number=Plur|PronType=Dem', 'Number=Sing', 'Number=Sing|Person=1|Poss=Yes|PronType=Prs', 'Number=Sing|Person=1|Poss=Yes|PronType=Prs|Reflex=Yes', 'Number=Sing|Person=Three|Tense=Pres|VerbForm=Fin', 'Number=Sing|PronType=Dem', 'Person=2|Poss=Yes|PronType=Prs', 'Person=2|PronType=Prs', 'Poss=Yes', 'Poss=Yes|PronType=Prs', 'PronType=Dem', 'PronType=Prs', 'PunctSide=Fin|PunctType=Brck', 'PunctSide=Fin|PunctType=Quot', 'PunctSide=Ini|PunctType=Brck', 'PunctSide=Ini|PunctType=Quot', 'PunctType=Comm', 'PunctType=Dash', 'PunctType=Peri', 'Tense=Past|VerbForm=Fin', 'Tense=Past|VerbForm=Part', 'Tense=Pres|VerbForm=Fin', 'VerbForm=Fin', 'VerbForm=Inf', 'VerbType=Mod', 'Gender=Neut|Number=Sing|Person=3|Poss=Yes|PronType=Prs|Reflex=Yes', 'Case=Nom|Gender=Neut|Number=Sing|Person=3|PronType=Prs', 'Case=Acc|Person=2|PronType=Prs', 'Case=Nom|Person=2|PronType=Prs', 'Case=Acc|Gender=Neut|Number=Sing|Person=3|PronType=Prs']
pos = ['PUNCT', 'ADV', 'AUX', 'SYM', 'ADP', 'SCONJ', 'VERB', 'X', 'PART', 'DET', 'NUM', 'NOUN', 'PRON', 'ADJ', 'CCONJ', 'PROPN', 'INTJ', 'SPACE']


n_splits = 500

if __name__=='__main__':
    
    parser = argparse.ArgumentParser(description="""""")
    parser.add_argument("--index", type=int, default=-1)
    parser.add_argument("--set", type=str)

    args = parser.parse_args()
    index = args.index
    file = files[0] if args.set=='train' else (files[1] if args.set=='test' else files[2])
    # Reading data
    data = open(file, 'r').read()
    n = len(data)
    print(n)
    # Splitting data
    data = data[index*n//n_splits: (index+1)*n//n_splits]
    # Computing in parallel
    text_to_doc(nlp, data, n_split_internal=500, name='_'+args.set+'_', index=index)
