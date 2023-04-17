import os
import glob
import time
import spacy
import numpy as np
import pandas as pd
import benepar
from tqdm import tqdm

from irnlm.utils import format_time, read_yaml

# Some formatting/abbreviation in LPP are not weel taken by the word list/freq dictionary and need to be corrected
correct = {'ve': 'have', 
           'hadn': 'had',
           'indulgently': 'indulgent',
           'abashed': 'confused',
           'sputtered': 'babbled',
           'seabird': 'bird',
           'gloomily': 'bitterly',
           'grumpily': 'cranky', 
           'panted': 'gasped', 
           'islet': 'isle', 
           'switchman': 'watchmaker', 
           'weathervane': 'anemometer', 
           'mustn': 'must' 
          }


def get_gutenberg_book_ids():
    """Return the list of Gutenberg bookds that
    were downloaded before behing filtered and cleaned.
    """
    result = open('data/gutenberg_books_list.txt', 'r').read().split(' ')
    return result

def load_path_from_ids(id_):
    """Load the path to the book associated with the id.
    Args:
        - id_: int
    Returns:
        - text: str
    """
    path = f'...{id_}'
    #text = open(path, 'r').read()
    #return text
    raise NotImplementedError('You need to create your own fetcher!')

def get_pronouns(language='english'):
    """Returns the list of pronouns for the chosen language.
    """
    if language=='english':
        result = open('data/english_pronouns.txt', 'r').read().split('\n')
    elif language=='french':
        result = open('data/french_pronouns.txt', 'r').read().split('\n')
    else:
        raise ValueError(f'Language {language} unknown.')
    return result

def get_negations(language='english'):
    """Returns the list of negations for the chosen language.
    """
    if language=='english':
        result = open('data/english_negations.txt', 'r').read().split('\n')
    elif language=='french':
        result = open('data/french_negations.txt', 'r').read().split('\n')
    else:
        raise ValueError(f'Language {language} unknown.')
    return result

def get_positions_words(language='english'):
    """Returns the list of positions for the chosen language.
    """
    if language=='english':
        result = open('data/english_positions.txt', 'r').read().split('\n')
    elif language=='french':
        result = open('data/french_positions.txt', 'r').read().split('\n')
    else:
        raise ValueError(f'Language {language} unknown.')
    return result

def get_quantity_words(language='english'):
    """Returns the list of quantity for the chosen language.
    """
    if language=='english':
        result = open('data/english_quantity.txt', 'r').read().split('\n')
    elif language=='french':
        result = open('data/french_quantity.txt', 'r').read().split('\n')
    else:
        raise ValueError(f'Language {language} unknown.')
    return result

def get_function_words(language='english'):
    """Returns the list of funtion words for the chosen language.
    """
    if language=='english':
        result = open('data/english_function_words.txt', 'r').read().split('\n')
    elif language=='french':
        result = open('data/french_function_words.txt', 'r').read().split('\n')
    else:
        raise ValueError(f'Language {language} unknown.')
    return result

def get_ids_syntax():
    """The syntax ids that are directly computed need to start from 0.
    This is what we correct here with the mapping that is being loaded.
    """
    result = read_yaml('data/syntax-id_to_train-id.yml')
    return result

def get_possible_morphs():
    """Retrieve list of possible TAGs.
    """
    result = ['AdvType=Ex', 'Aspect=Perf|Tense=Past|VerbForm=Part', 'Aspect=Prog|Tense=Pres|VerbForm=Part', 
              'Case=Acc|Gender=Fem|Number=Sing|Person=3|PronType=Prs', 'Case=Acc|Gender=Fem|Number=Sing|Person=3|PronType=Prs|Reflex=Yes', 
              'Case=Acc|Gender=Masc|Number=Sing|Person=3|PronType=Prs', 'Case=Acc|Gender=Masc|Number=Sing|Person=3|PronType=Prs|Reflex=Yes', 
              'Case=Acc|Gender=Neut|Number=Sing|Person=3|PronType=Prs|Reflex=Yes', 'Case=Acc|Number=Plur|Person=1|PronType=Prs', 
              'Case=Acc|Number=Plur|Person=1|PronType=Prs|Reflex=Yes', 'Case=Acc|Number=Plur|Person=3|PronType=Prs', 
              'Case=Acc|Number=Plur|Person=3|PronType=Prs|Reflex=Yes', 'Case=Acc|Number=Sing|Person=1|PronType=Prs', 
              'Case=Acc|Number=Sing|Person=1|PronType=Prs|Reflex=Yes', 'Case=Acc|Person=2|PronType=Prs|Reflex=Yes', 
              'Case=Nom|Gender=Fem|Number=Sing|Person=3|PronType=Prs', 'Case=Nom|Gender=Masc|Number=Sing|Person=3|PronType=Prs', 
              'Case=Nom|Number=Plur|Person=1|PronType=Prs', 'Case=Nom|Number=Plur|Person=3|PronType=Prs', 
              'Case=Nom|Number=Sing|Person=1|PronType=Prs', 'ConjType=Cmp', 'Definite=Def|PronType=Art', 
              'Definite=Ind|PronType=Art', 'Degree=Cmp', 'Degree=Pos', 'Degree=Sup', 'Foreign=Yes', 
              'Gender=Fem|Number=Sing|Person=3|Poss=Yes|PronType=Prs', 'Gender=Fem|Number=Sing|Person=3|Poss=Yes|PronType=Prs|Reflex=Yes', 
              'Gender=Masc|Number=Sing|Person=3|Poss=Yes|PronType=Prs', 'Gender=Masc|Number=Sing|Person=3|Poss=Yes|PronType=Prs|Reflex=Yes', 
              'Gender=Neut|Number=Sing|Person=3|Poss=Yes|PronType=Prs', 'Gender=Neut|Number=Sing|Person=3|PronType=Prs', 'Hyph=Yes', 
              'Mood=Ind|Number=Sing|Person=3|Tense=Past|VerbForm=Fin', 'Mood=Ind|Number=Sing|Person=3|Tense=Pres|VerbForm=Fin', 
              'Mood=Ind|Person=1|Tense=Pres|VerbForm=Fin', 'Mood=Ind|Tense=Pres|VerbForm=Fin', 'None', 'NounType=Prop|Number=Plur', 
              'NounType=Prop|Number=Sing', 'NumType=Card', 'NumType=Ord', 'Number=Plur', 'Number=Plur|Person=1|Poss=Yes|PronType=Prs', 
              'Number=Plur|Person=1|Poss=Yes|PronType=Prs|Reflex=Yes', 'Number=Plur|Person=2|Poss=Yes|PronType=Prs|Reflex=Yes', 
              'Number=Plur|Person=3|Poss=Yes|PronType=Prs', 'Number=Plur|Person=3|Poss=Yes|PronType=Prs|Reflex=Yes', 
              'Number=Plur|PronType=Dem', 'Number=Sing', 'Number=Sing|Person=1|Poss=Yes|PronType=Prs', 
              'Number=Sing|Person=1|Poss=Yes|PronType=Prs|Reflex=Yes', 'Number=Sing|Person=Three|Tense=Pres|VerbForm=Fin', 
              'Number=Sing|PronType=Dem', 'Person=2|Poss=Yes|PronType=Prs', 'Person=2|PronType=Prs', 'Poss=Yes', 
              'Poss=Yes|PronType=Prs', 'PronType=Dem', 'PronType=Prs', 'PunctSide=Fin|PunctType=Brck', 'PunctSide=Fin|PunctType=Quot', 
              'PunctSide=Ini|PunctType=Brck', 'PunctSide=Ini|PunctType=Quot', 'PunctType=Comm', 'PunctType=Dash', 'PunctType=Peri', 
              'Tense=Past|VerbForm=Fin', 'Tense=Past|VerbForm=Part', 'Tense=Pres|VerbForm=Fin', 'VerbForm=Fin', 'VerbForm=Inf', 
              'VerbType=Mod', 'Gender=Neut|Number=Sing|Person=3|Poss=Yes|PronType=Prs|Reflex=Yes', 
              'Case=Nom|Gender=Neut|Number=Sing|Person=3|PronType=Prs', 'Case=Acc|Person=2|PronType=Prs', 'Case=Nom|Person=2|PronType=Prs', 
              'Case=Acc|Gender=Neut|Number=Sing|Person=3|PronType=Prs']
    return result

def get_possible_tag():
    """Retrieve list of possible TAGs.
    """
    result = [
        '$', "''", ',', '-LRB-', '-RRB-', '.', ':', 
        'ADD', 'AFX', 'CC', 'CD', 'DT', 'EX', 'FW', 
        'HYPH', 'IN', 'JJ', 'JJR', 'JJS', 'LS', 'MD', 
        'NFP', 'NN', 'NNP', 'NNPS', 'NNS', 'PDT', 
        'POS', 'PRP', 'PRP$', 'RB', 'RBR', 'RBS', 
        'RP', 'SYM', 'TO', 'UH', 'VB', 'VBD', 'VBG', 
        'VBN', 'VBP', 'VBZ', 'WDT', 'WP', 'WP$', 
        'WRB', 'XX', '``'
        ]
    return result

def get_possible_pos():
    """Retrieve list of possible POS.
    """
    result = ['PUNCT', 'ADV', 'AUX', 'SYM', 'ADP', 
              'SCONJ', 'VERB', 'X', 'PART', 'DET', 
              'NUM', 'NOUN', 'PRON', 'ADJ', 'CCONJ', 
              'PROPN', 'INTJ', 'SPACE']
    return result

def get_possible_dep():
    """Retrieve list of possible dependency relations.
    """
    result = [
        'ROOT', 'acl', 'acomp', 'advcl', 'advmod', 
        'agent', 'amod', 'appos', 'attr', 'aux', 
        'auxpass', 'case', 'cc', 'ccomp', 'compound', 
        'conj', 'csubj', 'csubjpass', 'dative', 'dep', 
        'det', 'dobj', 'expl', 'intj', 'mark', 'meta', 
        'neg', 'nmod', 'npadvmod', 'nsubj', 
        'nsubjpass', 'nummod', 'oprd', 'parataxis', 
        'pcomp', 'pobj', 'poss', 'preconj', 'predet', 
        'prep', 'prt', 'punct', 'quantmod', 'relcl',
        'xcomp'
        ]
    return result

def get_punctuation():
    """Return list of punctuation signs.
    """
    result = [
        '\'', ',', ';', ':', '/', '-', '"', '‘', 
        '’', '(', ')', '{', '}', '[', ']', '`', 
        '“', '”', '—', '.', '!', '?', '...'
        ]
    return result

def get_word_list_and_freq(path='data/lexique_database.tsv'):
    """Retrieve list of english words and their frequenc in
    a given database.
    """
    english_words_data = pd.read_csv(path, delimiter='\t')
    # Creating dict with freq information
    word_list = english_words_data['Word'].apply(lambda x: str(x).lower()).values
    freq_list = english_words_data['Lg10WF'].values
    zip_freq = zip(word_list, freq_list)
    word_freq = dict(zip_freq)
    return word_list, word_freq

def set_nlp_pipeline(name="en_core_web_lg", to_remove=['ner'], max_length=np.inf):
    """Load and prepare Spacy NLP pipeline.
    """
    nlp = spacy.load(name)
    for pipe in to_remove:
        nlp.remove_pipe(pipe)
    nlp.max_length = max_length
    print(nlp.pipe_names)
    return nlp

def extract_pos_info(token, children_max_size=5):
    """Extract syntactic POS information.
    Args:
        - token: spacy token object
    Returns:
        - list
    """
    children_pos = [child.pos_ for child in token.children]
    children_pos = children_pos[:5] if len(children_pos)>=5 else (children_pos + ['' for i in range(5-len(children_pos))])
    all_pos = [token.pos_, token.head.pos_] + children_pos
    return '-'.join(all_pos)

def extract_pipeline_info(token, ner=False):
    """Extract information from preprocess input sentence.
    Args:
        - token: spacy token object
    Returns:
        - list
    """
    if ner:
        return [token.text.lower(), token.lemma_, 
                token.pos_, token.tag_, token.dep_, 
                token.head.text.lower(), token.head.pos_, 
                token.head.tag_, token.shape_, str(token.morph) if str(token.morph)!='' else 'None', 
                [child.pos_ for child in token.children], 
                [child.dep_ for child in token.children], 
                [child.tag_ for child in token.children],
                token.ent_iob_, token.ent_type_, token.ent_kb_id_]
    else:
        return [token.text.lower(), token.lemma_, 
                token.pos_, token.tag_, token.dep_, 
                token.head.text.lower(), token.head.pos_, 
                token.head.tag_, token.shape_, str(token.morph), 
                [child.pos_ for child in token.children], 
                [child.dep_ for child in token.children], 
                [child.tag_ for child in token.children]]

def apply_pipeline(nlp, sentence):
    """Apply Spacy pipeline to a sentence and returns a dataframe containing
    parsed information.
    """
    print('Processing...')
    t0 = time.time()
    doc = nlp(sentence)
    print(f'Processed in {format_time(time.time() - t0)}.')
    data = []
    columns=['text', 'lemma', 'pos', 'tag', 'dep', 'head', 'head_pos', 'head_tag', 'shape', 'morph', 'children_pos', 'children_dep', 'children_tag']
    if 'ner' in nlp.pipe_names:
        columns += ['entity_IOB', 'entity_type', 'entity']
    #data = Parallel(n_jobs=-1)(delayed(extract_pipeline_info)(token, 'ner' in nlp.pipe_names) for index, token in tqdm(enumerate(doc)))
    for index, token in enumerate(doc):
        if 'ner' in nlp.pipe_names:
            data.append([
                token.text.lower(), token.lemma_, 
                token.pos_, token.tag_, token.dep_, 
                token.head.text.lower(), token.head.pos_, 
                token.head.tag_, token.shape_, str(token.morph) if str(token.morph)!='' else 'None', 
                [child.pos_ for child in token.children], 
                [child.dep_ for child in token.children], 
                [child.tag_ for child in token.children],
                token.ent_iob_, token.ent_type_, token.ent_kb_id_])
        else:
            data.append([
                token.text.lower(), token.lemma_, 
                token.pos_, token.tag_, token.dep_, 
                token.head.text.lower(), token.head.pos_, 
                token.head.tag_, token.shape_, str(token.morph), 
                [child.pos_ for child in token.children], 
                [child.dep_ for child in token.children], 
                [child.tag_ for child in token.children]
            ])
    df = pd.DataFrame(data, columns=columns)
    return df

def doc_to_tree(doc, root, tree_type='pos'):
    """Create linear tree structure to represent dependency parsing.
    """
    representation = [root.pos_ if tree_type=='pos' else root.tag_]
    tmp = []
    for descendant in root.children:
        tmp.append(doc_to_tree(doc, descendant))
    representation += tmp
    return representation

def add_constituent_parser(nlp):
    """Add constituent parser to Spacy NLP parser.
    """
    try:
        nlp.add_pipe("benepar", config={"model": "benepar_en3"})
    except:
        nlp.add_pipe(benepar.BeneparComponent("benepar_en3"))
    return nlp

def get_constituent_parent(span):
    """Get parent tokens in the constituency parsing.
    """
    parents = span if span._.parent is None else span._.parent
    return parents

def possible_subjects_id(language):
    """ Returns possible subject id list for a given language.
    Arguments:
        - language: str
    Returns:
        result: list (of int)
    """
    if language=='english':
        result = [57, 58, 59, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70,
                    72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 86, 87, 88, 89, 91, 92, 93,
                    94, 95, 96, 97, 98, 99, 100, 101, 103, 104, 105, 106, 108, 109, 110, 113, 114, 115]
    elif language=='french':
        result = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,22,23,24,25,26,27,29,30] # ,54,55,56,57,58,59,61,62,63,64,65 # TO DO #21 was removed issue to investigate
    elif language=='chineese':
        result = [1] # TO DO
    elif language=='ibc':
        result = ['FR065', 'FR064', 'FR063', 'FR062', 'FR061', 'FR059', 'FR058', 'FR057', 'FR056', 'FR055', 'FR054']
    else:
        raise Exception('Language {} not known.'.format(language))
    return result

def get_nscans(language):
    """Returns the number of scans per session for a given language.
    Args:
        - language: str
    Returns:
        - int
    """
    result = {'english':{'run1':282,
                             'run2':298,
                             'run3':340,
                             'run4':303,
                             'run5':265,
                             'run6':343,
                             'run7':325,
                             'run8':292,
                             'run9':368
        },
                       'french':{'run1':309,
                             'run2':326,
                             'run3':354,
                             'run4':315,
                             'run5':293,
                             'run6':378,
                             'run7':332,
                             'run8':294,
                             'run9':336
        }
        }
    return result[language]

def get_subject_name(id):
    """ Get subject name from id.
    Arguments:
        - id: int
    Returns:
        - str
    """
    if type(id)==str:
        return 'sub-{}'.format(id)
    elif id < 10:
        return 'sub-00{}'.format(id)
    elif id < 100:
        return 'sub-0{}'.format(id)
    else:
        return 'sub-{}'.format(id)
    
def fetch_paths(path, name):
    """Concatenate a path and a template name and search for matches.
    Args:
        - path: str
        - name: str
    Returns:
        - files: list (str)
    """
    path = os.path.join(path, name)
    files = sorted(glob.glob(path))
    return files

def fetch_model_maps(
        model_names, 
        language='english',
        verbose=1,
        FMRIDATA_PATH="derivatives/fMRI/maps/english"
    ):
    """Retrieve the maps computed for each subject of The Little Prince.
    Arguments:
        - model_names: list of string
        - language: string
        - verbose: int
        - FMRIDATA_PATH: string, path to fMRI data
    Returns:
        - data_full: dicitonary where 'model_names' are keys and values are also dictionary
           e.g.: data_full = {'glove': {'R2': [path_to_map_sub-1, path_to_map_sub-2, ...], 
                                        'Pearson_coeff': [path_to_map_sub-1, path_to_map_sub-2, ...]
                                        }
                             }
    """
    data_full = {}
    for model_name in model_names:
        data_full[model_name.replace('_{}', '')] = {}
        R2_maps = {}
        Pearson_coeff_maps = {}
        baseline_R2_maps = {}
        baseline_Pearson_coeff_maps = {}
        for subject_id in tqdm(possible_subjects_id(language)):
            subject = get_subject_name(subject_id)
            path_to_map = os.path.join(FMRIDATA_PATH, subject, model_name.format(subject_id))
            R2_maps[subject] = fetch_paths(path_to_map, '*R2.nii.gz')
            R2_maps[subject] = [i for i in R2_maps[subject] if 'baseline' not in i]
            Pearson_coeff_maps[subject] = fetch_paths(path_to_map, '*Pearson_coeff.nii.gz')
            Pearson_coeff_maps[subject] = [i for i in Pearson_coeff_maps[subject] if 'baseline' not in i]
            try:
                baseline_R2_maps[subject] = fetch_paths(path_to_map, '*baseline_R2.nii.gz')
                baseline_Pearson_coeff_maps[subject] = fetch_paths(path_to_map, '*baseline_Pearson_coeff.nii.gz')
            except:
                print(f'No baseline for {model_name}.')
                baseline_R2_maps[subject] = []
                baseline_Pearson_coeff_maps[subject] = []
            if verbose > 0:
                print(subject, '-', len(R2_maps[subject]), '-', len(Pearson_coeff_maps[subject]), '-', len(baseline_R2_maps[subject]), '-', len(baseline_Pearson_coeff_maps[subject]))
        R2_lists = list(zip(*R2_maps.values()))
        Pearson_coeff_lists = list(zip(*Pearson_coeff_maps.values()))
        try:
            baseline_R2_lists = list(zip(*baseline_R2_maps.values()))
            baseline_Pearson_coeff_lists = list(zip(*baseline_Pearson_coeff_maps.values()))
        except:
            R2_lists = []
            Pearson_coeff_lists = []
        try:
            data_full[model_name.replace('_{}', '')] = {
                'R2': list(R2_lists[0]) if len(R2_lists)>0 else [],
                'Pearson_coeff': list(Pearson_coeff_lists[0]) if len(Pearson_coeff_lists)>0 else [],
                'baseline_R2': list(baseline_R2_lists[0]) if len(baseline_R2_lists)>0 else [],
                'baseline_Pearson_coeff': list(baseline_Pearson_coeff_lists[0]) if len(baseline_Pearson_coeff_lists)>0 else [],
                                }
        except:
            print('ERROR with: ', model_name)
    return data_full
