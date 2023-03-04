import os, glob
import numpy as np
import pandas as pd
from tqdm import tqdm

import benepar
from spacy.symbols import ORTH
from nltk.corpus import wordnet as wn
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import StratifiedGroupKFold, ShuffleSplit, StratifiedShuffleSplit, StratifiedKFold, GroupKFold

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from mpl_toolkits.axes_grid.parasite_axes import SubplotHost

from irnlm.data.utils import (
    set_nlp_pipeline, 
    get_ids_syntax, 
    get_function_words, 
    get_punctuation, 
    get_possible_morphs, 
    get_possible_pos
)
from irnlm.data.extract_syntactic_features import extract_syntax
from irnlm.utils import check_folder



def get_category(word):
    try:
        name = wn.synsets(word)[0].name()
        category = wn.synset(name).hypernyms()[0].name()
        res = categories[category.split('.')[0]]
    except:
        res = categories['None']
    return res


saving_folder = 'derivatives/Fig2/'
check_folder(saving_folder)

nlp = set_nlp_pipeline()
special_case = [{ORTH: "hasnt"}]
nlp.tokenizer.add_special_case("hasnt", special_case)
benepar.download('benepar_en3')
nlp.add_pipe('benepar', config={'model': 'benepar_en3'})
print(nlp.pipe_names)



tokenizer_dict = {
    '<s>': 0, '<pad>': 1, '</s>': 2, '<unk>': 3, '<mask>': 4, 'PUNCT': 5, 'ADV': 6, 
    'AUX': 7, 'SYM': 8, 'ADP': 9, 'SCONJ': 10, 'VERB': 11, 'X': 12, 'PART': 13, 
    'DET': 14, 'NUM': 15, 'NOUN': 16, 'PRON': 17, 'ADJ': 18, 'CCONJ': 19, 'PROPN': 20, 
    'INTJ': 21, 'SPACE': 22
}
#for i, iterator in enumerate(iterator_list):
#    X_test = iterator
#    X_train = [*iterator_list[:i], *iterator_list[i+1:]] if ((len(iterator_list[:i])!=0) and (len(iterator_list[i+1:])!=0)) else (iterator_list[:i] if (len(iterator_list[i+1:])==0) else iterator_list[i+1:])
#    X_train = [i for l in X_train for i in l]
#    iterator = [item.strip() for item in iterator]
#    iterator = ' '.join(iterator)
#    doc = nlp(iterator)
#    iterator = []
#    tokenized_text = []
#    for token in tqdm(doc):
#        tokenized_text.append(token.pos_)
#        iterator.append(tokenizer_dict[token.pos_])
#    mapping = {i:[i] for i, j in enumerate(iterator)}

folder = './'

X_gpt2_syntax = [pd.read_csv(f).values for f in sorted(glob.glob('data/stimuli-representations/english/GPT-2_L-4_H-768_Syntax_context-full_end-epoch-4_split-3/act*.csv'))]
X_gpt2_semantic = [pd.read_csv(f).values for f in sorted(glob.glob('data/stimuli-representations/english/GPT-2_L-4_H-768_semantic_context-full_end-epoch-4_split-3/act*.csv'))]
X_gpt2 = [pd.read_csv(f).values for f in sorted(glob.glob('data/stimuli-representations/english/GPT-2_L-4_H-768_default_tokenizer_full_end-epoch-4_split-3/act*.csv'))]

X_Glove_syntax = [pd.read_csv(f).values for f in sorted(glob.glob('data/stimuli-representations/english/GloVe_Syntax_full/act*.csv'))]
X_Glove_semantic = [pd.read_csv(f).values for f in sorted(glob.glob('data/stimuli-representations/english/GloVe_Semantic/act*.csv'))]
X_Glove = [pd.read_csv(f).values for f in sorted(glob.glob('data/stimuli-representations/english/GloVe/act*.csv'))]

Y_syntax = [list(pd.read_csv(f)['word'].values) for f in sorted(glob.glob('data/onsets-offsets/english/syntax_run*'))]
Y_semantic = [list(pd.read_csv(f)['word'].values) for f in sorted(glob.glob('data/onsets-offsets/english/semantic_run*'))]

Y = [list(pd.read_csv(f)['word'].values) for f in sorted(glob.glob('data/onsets-offsets/english/word+punctuation_run*'))]

function_words = get_function_words()
punctuation = get_punctuation()
morphs = get_possible_morphs()
pos = get_possible_pos()

all_pos = []
all_syntax = []
all_morph = []
all_ncn = []

for i, iterator in enumerate(Y):
    doc = nlp(' '.join(iterator))
    iterator = []
    iterator, pos_, morph, ncn = extract_syntax(doc)
    #for token in doc:
    #    iterator.append(str(token.pos_)+str(token.head.pos_)+(str(token.morph) if str(token.morph)!='' else 'None'))
    all_pos.append(pos_)
    all_syntax.append(iterator)
    all_morph.append(morph)
    all_ncn.append(ncn)
    
pos_tmp = [i for l in all_pos for i in l]
morph_tmp = [i for l in all_morph for i in l]
ncn_tmp = [i for l in all_ncn for i in l]
syntax_tmp = [i for l in all_syntax for i in l]


categories = []
for i in range(9):
    for word in Y[i]:
        if word not in function_words:
            for k, j in enumerate(wn.synsets(word)):
                if k ==0:
                    name = j.name()
            try:
                categories.append(wn.synset(name).hypernyms()[0])
                print(word, '-', categories[-1])
            except:
                _=0

categories = {name: i for i, name in enumerate(list(set([i.name().split('.')[0] for i in list(set(categories))])))}
categories['None'] = len(categories)
print(len(categories))


results_syntax = {}
model_names = ['Glove_syntax', 'Glove_semantic', 'Glove', 'GPT2_syntax', 'GPT2_semantic', 'GPT2'] # 'Baseline',

data_type_names = ['Syntax'] #'POS', 'Morph', 'NCN', 

for k, X_tmp in enumerate([X_Glove_syntax, X_Glove_semantic, X_Glove, X_gpt2_syntax, X_gpt2_semantic, X_gpt2]): 
    groups = [i.shape[0] for i in X_tmp]
    groups = [[i]*j  for i, j in enumerate(groups)]
    groups = [i for j in groups for i in j]

    results_syntax[model_names[k]] = {}
    for j, data_to_predict in tqdm(enumerate([all_syntax])): # all_pos, all_morph, all_ncn, 
        results_syntax[model_names[k]][data_type_names[j]] = []
        unique = {name: i for i, name in enumerate(np.unique([i for l in data_to_predict for i in l]))}
        
        #sss = StratifiedShuffleSplit(n_splits=9, test_size=0.1, random_state=1111)
        sss = StratifiedGroupKFold(n_splits=9)
        #sss = StratifiedKFold(n_splits=9)
        #sss = ShuffleSplit(n_splits=10, test_size=0.1, random_state=1111)
        
        x = np.vstack(X_tmp)
        if 'semantic' in model_names[k]:
            y = [unique[n] for m, l in enumerate(data_to_predict) for k, n in enumerate(l) if Y[m][k] not in function_words]
        else:
            y = [unique[n] for l in data_to_predict for n in l]
        y = np.array(y)
        
        for train_index, test_index in sss.split(x, y, groups=groups):
            X_train, X_test = x[train_index], x[test_index]
            Y_train, Y_test = y[train_index], y[test_index]
            #print(X_train, X_test)

            #clf = RidgeClassifier().fit(X_train, Y_train)
            clf = DummyClassifier(strategy="most_frequent", random_state=1111).fit(X_train, Y_train)
            #clf = LogisticRegression(random_state=1111).fit(X_train, Y_train)
            results_syntax[model_names[k]][data_type_names[j]].append(clf.score(X_test, Y_test))
print(results_syntax)


results_semantic = {}
model_names = ['Glove_syntax', 'Glove_semantic', 'Glove', 'GPT2_syntax', 'GPT2_semantic', 'GPT2'] # 'Baseline', 
#model_names = [ 'Baseline_syntax']
for k, X_tmp in tqdm(enumerate([X_Glove_syntax, X_Glove_semantic, X_Glove, X_gpt2_syntax, X_gpt2_semantic, X_gpt2])):
    print(k)
    groups = [i.shape[0] for i in X_tmp]
    groups = [[i]*j  for i, j in enumerate(groups)]
    if 'semantic' not in model_names[k]:
        groups = [n for m, l in enumerate(groups) for r, n in enumerate(l) if Y[m][r] not in function_words]
    else:
        groups = [i for j in groups for i in j]
    
    results_semantic[model_names[k]] = []

    #sss = StratifiedShuffleSplit(n_splits=9, test_size=0.1, random_state=1111)
    sss = StratifiedGroupKFold(n_splits=9)
    #sss = ShuffleSplit(n_splits=10, test_size=0.1, random_state=1111)

    x = np.vstack(X_tmp)
    y = [get_category(n) for m, l in enumerate(Y) for k, n in enumerate(l) if Y[m][k] not in function_words]
    if 'semantic' not in model_names[k]:
        tmp = [j for l in Y for j in l]
        x = np.vstack([x[i, :] for i in range(x.shape[0]) if tmp[i] not in function_words])
        #y = [get_category(n) for l in Y for n in l]
    y = np.array(y)
    for train_index, test_index in sss.split(x, y, groups=groups):
        X_train, X_test = x[train_index], x[test_index]
        Y_train, Y_test = y[train_index], y[test_index]
        #print(X_train, X_test)

        #clf = RidgeClassifier().fit(X_train, Y_train)
        #clf = LogisticRegression(random_state=1111).fit(X_train, Y_train)
        clf = DummyClassifier(strategy="most_frequent", random_state=1111).fit(X_train, Y_train)

        results_semantic[model_names[k]].append(clf.score(X_test, Y_test))
print(results_semantic)



fig1 = plt.figure(figsize=(22, 5))
ax1 = SubplotHost(fig1, 111)
fig1.add_subplot(ax1)


################################################################
################################################################

#plt.rcParams['font.size'] = 30
### Set the default text font size
##plt.rc('font', size=30)
### Set the axes title font size
#plt.rc('axes', titlesize=20)
### Set the axes labels font size
#plt.rc('axes', labelsize=20)
### Set the font size for x tick labels
#plt.rc('xtick', labelsize=30)
### Set the font size for y tick labels
#plt.rc('ytick', labelsize=30)
### Set the legend font size
##plt.rc('legend', fontsize=25)
### Set the font size of the figure title
#plt.rc('figure', titlesize=20)
#
################################################################
################################################################

#################################################################################################################

to_plot_syntax = {}
for i, model in enumerate(results_syntax.keys()):
    for key in results_syntax[model].keys():
        #minimum = np.mean(results_syntax['Baseline'][key])
        minimum = 0.14937310146915872 if 'semantic' in model else 0.06467317953681476
        if key in to_plot_syntax.keys():
            #to_plot_syntax[key].append((np.array(results_syntax[model][key])-minimum)/(1-minimum))
            to_plot_syntax[key].append(results_syntax[model][key])
        else:
            #to_plot_syntax[key] = [(np.array(results_syntax[model][key])-minimum)/(1-minimum)]
            to_plot_syntax[key] = [results_syntax[model][key]]
            
for key in to_plot_syntax.keys():
    plt.boxplot(to_plot_syntax[key])
    plt.title(key, fontsize=30)
    plt.xticks(np.arange(1, len(model_names)+1), model_names, rotation=90, fontsize=20)
    plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1],[0, 0.2, 0.4, 0.6, 0.8, 1], fontsize=20)
    format_figure = 'pdf'
    dpi='100'
    plot_name = f'Probing_{key}_in_model_activations'
    plt.savefig(os.path.join(saving_folder, f'{plot_name}.{format_figure}'), format=format_figure, dpi=dpi, bbox_inches = 'tight', pad_inches = 0, )
    plt.show()
    
#################################################################################################################
to_plot_semantic = []
for i, model in enumerate(results_semantic.keys()):
    #minimum = np.mean(results_semantic['Baseline'])
    minimum = 0.1500998728843726
    #to_plot_semantic.append((np.array(results_semantic[model])-minimum)/(1-minimum))
    to_plot_semantic.append(results_semantic[model])

plt.boxplot(to_plot_semantic)
plt.title('Semantic', fontsize=30)
plt.xticks(np.arange(1, len(model_names)+1), model_names, rotation=90, fontsize=20)
plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1],[0, 0.2, 0.4, 0.6, 0.8, 1], fontsize=20)
format_figure = 'pdf'
dpi='100'
plot_name = 'Probing_Semantic_in_model_activations'
#plt.savefig(os.path.join(saving_folder, f'{plot_name}.{format_figure}'), format=format_figure, dpi=dpi, bbox_inches = 'tight', pad_inches = 0, )
plt.show()

#################################################################################################################




names = [model_names[i] for i in [1, 4, 2, 5, 3, 6]]
names = ['GloVe', 'GPT-2', ''] * 3
names = names[:-1]
to_plot_syntax_ = [to_plot_syntax['Syntax'][i-1] for i in [1, 4, 2, 5, 3, 6]]
to_plot_syntax_ = to_plot_syntax_[:2] + [np.zeros(9)] + to_plot_syntax_[2:4] + [np.zeros(9)] + to_plot_syntax_[4:6] 
to_plot_semantic_ = [to_plot_semantic[i-1] for i in [1, 4, 2, 5, 3, 6]]
to_plot_semantic_ = to_plot_semantic_[:2] + [np.zeros(9)] + to_plot_semantic_[2:4] + [np.zeros(9)] + to_plot_semantic_[4:6] 

#fig = plt.figure(figsize=(10, 10))
ind = np.arange(1, 1+len(names))
width = 0.35
ax1.barh(ind - width/2, [np.mean(i) for i in to_plot_semantic_], width, yerr=[np.std(i) for i in to_plot_semantic_],
                label='Semantic task', color='green', align='center')
ax1.barh(ind + width/2, [np.mean(i) for i in to_plot_syntax_], width, yerr=[np.std(i) for i in to_plot_syntax_],
                label='Syntactic task', color='red', align='center')

# Baselines
baselines_syntax_ = [B_syntax[i-1] for i in [1, 4, 2, 5, 3, 6]]
baselines_syntax_ = baselines_syntax_[:2] + [np.zeros(9)] + baselines_syntax_[2:4] + [np.zeros(9)] + baselines_syntax_[4:6] 
baselines_semantic_ = [B_semantic[i-1] for i in [1, 4, 2, 5, 3, 6]]
baselines_semantic_ = baselines_semantic_[:2] + [np.zeros(9)] + baselines_semantic_[2:4] + [np.zeros(9)] + baselines_semantic_[4:6] 

ax1.barh(ind - width/2, [np.mean(i) for i in baselines_semantic_], width, yerr=[np.std(i) for i in baselines_semantic_],
                color=(0.1, 0.1, 0.1, 0.1), align='center', alpha=0.1, edgecolor='black', linewidth=1)
ax1.barh(ind + width/2, [np.mean(i) for i in baselines_syntax_], width, yerr=[np.std(i) for i in baselines_syntax_],
                color=(0.1, 0.1, 0.1, 0.1), align='center', alpha=0.1, edgecolor='black', linewidth=1)
#
#ax1.barh(ind - width/2, [np.mean(i) for i in baselines_semantic_], width, yerr=[np.std(i) for i in baselines_semantic_],
#                color='green', align='center', alpha=1, edgecolor='black', linewidth=0)
#ax1.barh(ind + width/2, [np.mean(i) for i in baselines_syntax_], width, yerr=[np.std(i) for i in baselines_syntax_],
#                label='Baseline', color='red', align='center', alpha=1, edgecolor='black', linewidth=0)


ax1.set_yticks(np.arange(1, len(names)+1))
ax1.set_yticklabels(names)
#
#ax1.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1])
#ax1.set_xticklabels([0, 0.2, 0.4, 0.6, 0.8, 1])

#ax1.tick_params(axis='x', which='major', labelsize=30)
#ax1.tick_params(axis='y', which='major', rotation=0, labelsize=30)

plt.setp(ax1.get_xticklabels(), rotation=0, ha="left", fontsize=30, size=30)

ax1.spines['right'].set_visible(False)
ax1.spines['top'].set_visible(False)


format_figure = 'pdf'
dpi='300'
plt.legend(loc='middle right',
           ncol=1,
           fontsize=25)
ax1.invert_yaxis()


ax2 = ax1.twinx()
ax2.spines['right'].set_visible(False)
ax2.spines['top'].set_visible(False)

offset = -100, 0 # Position of the second axis
new_axisline = ax2.get_grid_helper().new_fixed_axis
ax2.axis["left"] = new_axisline(loc="left", axes=ax2, offset=offset)
ax2.set_yticks([6, 4, 2, 0], minor=False)
ax2.set_yticks([5, 3, 1], minor=True)

#ax2.set_yticklabels(['Original', 'Semantic', 'Syntax'])
#ax2.set_yticklabels(['', 'Syntax', '', 'Semantic', '', 'Original'])
#plt.rc('ytick', labelsize=30)
ax2.set_yticklabels([ 'Syntax', 'Semantic',  'Original'], 
                    {'fontsize': 25}, minor=True)
#plt.rc('ytick', labelsize=25)
ax2.set_yticklabels([ '', '',  '', ''], 
                    {'fontsize': 20}, minor=False)

#plt.rc('xtick', labelsize=30)
plt.xlabel('Decoding Accuracy', fontsize=30)
#ax2.set_yticks([4,2])



plot_name = 'Probing_Syntax-Semantic_in_model_activations_with_baseline'
#plt.tight_layout()
plt.savefig(os.path.join(saving_folder, f'{plot_name}.{format_figure}'), format=format_figure, dpi=dpi, bbox_inches = 'tight', pad_inches = 0, )
plt.show()

plt.close('all')