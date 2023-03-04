import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from irnlm.utils import check_folder
from fmri_encoder.data import fetch_masker
from fmri_encoder.plotting import plot_colorbar
from irnlm.contrasts import plot_brain_fit


red = plt.cm.get_cmap('black_red')
red = red(np.linspace(0, 1, 350))
red = ListedColormap(red[0:260])
plot_colorbar(red)

green = np.array([[i/50, 1, 0, 1] for i in range(50)])
green = np.vstack([np.array([[0, i/100, 0, 1] for i in range(100)]), green, np.array([[1, 1, i/25, 1] for i in range(25)])])
green = ListedColormap(np.vstack([green]))
plot_colorbar(green, 0.05, 0)

blue = np.array([[ 0, i/50, 1, 1] for i in range(50)])
blue = np.vstack([ np.array([[0, 0, i/100, 1] for i in range(100)]), blue, np.array([[i/25, 1, 1, 1] for i in range(25)])])
blue = ListedColormap(np.vstack([blue]))
plot_colorbar(blue, 0.1, 0)

if __name__=='__main__':

    ###
    # Replicating Fig5: Unique Correlations
    ###
    saving_folder = 'derivatives/Fig5'
    check_folder(saving_folder)
    masker = fetch_masker('derivatives/masker', None, **{'detrend': False, 'standardize': False})

    # GloVe Syntax
    name0 = "GloVe_Syntax_full_GloVe_Semantic_cv-alpha_-3_to_4wordrate-rms_chris-log_frequency_{}"
    name1 = "GloVe_Semanticcv-alpha_-3_to_4wordrate-rms_chris-log_frequency_{}"
    effmaps_dict, zmaps_dict = plot_brain_fit(
        name0, name1, vmax=0.06, cmap=red, surname='Glove_Syntax+Semantic-vs-Glove_Semantic', 
        masker=masker, height_control='fdr', alpha=0.005, saving_folder=saving_folder)
    
    # GloVe Semantic
    name0 = "GloVe_Syntax_full_GloVe_Semantic_cv-alpha_-3_to_4wordrate-rms_chris-log_frequency_{}"
    name1 = "GloVe_Syntax_fullcv-alpha_-3_to_4wordrate-rms_chris-log_frequency_{}"
    effmaps_dict, zmaps_dict = plot_brain_fit(
        name0, name1, vmax=0.06, cmap=green, surname='Glove_Syntax+Semantic-vs-Glove_Syntax', 
        masker=masker, height_control='fdr', alpha=0.005, saving_folder=saving_folder)
    
    # GPT-2 Syntax
    name0 = "GPT-2_L-4_H-768_Syntax_context-full_end-epoch-4_split-3-GPT-2_L-4_H-768_semantic_context-full_end-epoch-4_split-3_norm-None_temporal-shifting-0_{}_hidden-layer-3cv-alpha_-3_to_4_no-centering_wordrate-rms_chris-log_frequency"
    name1 = "GPT-2_L-4_H-768_semantic_context-full_end-epoch-4_split-3_norm-None_temporal-shifting-0_{}_hidden-layer-3cv-alpha_-3_to_4_no-centering_wordrate-rms_chris-log_frequency"
    effmaps_dict, zmaps_dict = plot_brain_fit(
        name0, name1, vmax=0.06, cmap=red, surname='GPT-2_Syntax+Semantic-vs-GPT-2_Semantic', 
        masker=masker, height_control='fdr', alpha=0.005, saving_folder=saving_folder)
    
    # GPT-2 Semantic
    name0 = "GPT-2_L-4_H-768_Syntax_context-full_end-epoch-4_split-3-GPT-2_L-4_H-768_semantic_context-full_end-epoch-4_split-3_norm-None_temporal-shifting-0_{}_hidden-layer-3cv-alpha_-3_to_4_no-centering_wordrate-rms_chris-log_frequency"
    name1 = "GPT-2_L-4_H-768_Syntax_context-full_end-epoch-4_split-3_norm-None_temporal-shifting-0_{}_hidden-layer-3cv-alpha_-3_to_4_no-centering_wordrate-rms_chris-log_frequency"
    effmaps_dict, zmaps_dict = plot_brain_fit(
        name0, name1, vmax=0.06, cmap=green, surname='GPT-2_Syntax+Semantic-vs-GPT-2_Syntax', 
        masker=masker, height_control='fdr', alpha=0.005, saving_folder=saving_folder)

    # GloVe Synergy
    name0 = "GloVecv-alpha_-3_to_4wordrate-rms_chris-log_frequency_{}"
    name1 = "GloVe_Syntax_full_GloVe_Semantic_cv-alpha_-3_to_4wordrate-rms_chris-log_frequency_{}"
    effmaps_dict, zmaps_dict = plot_brain_fit(
        name0, name1, vmax=0.1, cmap=green, surname='GloVe_full-vs-GloVe_Sem+Syn_intertwined', 
        masker=masker, height_control='fdr', alpha=0.005, saving_folder=saving_folder)

    # GPT-2 Synergy
    name0 = "GPT-2_L-4_H-768_default_tokenizer_full_end-epoch-4_split-3_norm-None_temporal-shifting-0_{}_hidden-layer-3cv-alpha_-3_to_4_no-centering_wordrate-rms_chris-log_frequency"
    name1 = "GPT-2_L-4_H-768_Syntax_context-full_end-epoch-4_split-3-GPT-2_L-4_H-768_semantic_context-full_end-epoch-4_split-3_norm-None_temporal-shifting-0_{}_hidden-layer-3cv-alpha_-3_to_4_no-centering_wordrate-rms_chris-log_frequency"
    effmaps_dict, zmaps_dict = plot_brain_fit(
        name0, name1, vmax=0.1, cmap=blue, surname='GPT-2_full-vs-GPT-2_Sem+Syn_intertwined', 
        masker=masker, height_control='fdr', alpha=0.005, saving_folder=saving_folder)
