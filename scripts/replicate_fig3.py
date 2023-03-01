import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from irnlm.utils import check_folder
from fmri_encoder.data import fetch_masker
from fmri_encoder.plotting import plot_colorbar
from irnlm.contrasts import (
    compute_diff_group_level_maps, 
    plot_group_level_surface_maps
)
from irnlm.peak_regions import plot_peak_regions

green = np.array([[i/50, 1, 0, 1] for i in range(50)])
green = np.vstack([np.array([[0, i/100, 0, 1] for i in range(100)]), green, np.array([[1, 1, i/25, 1] for i in range(25)])])
green = ListedColormap(np.vstack([green]))
plot_colorbar(green, 0.05, 0)

def plot_brain_fit(
        name0, 
        name1, 
        vmax=0.1, 
        cmap='cold_hot', 
        surname=None, 
        masker=None, 
        height_control='fdr', 
        alpha=0.005,
        saving_folder=None,
    ):
    """Compute the group level difference between model `name0` and model `name1`.
    Return significant values and correct for multiple comparison with a FDR 
    correction of 0.005.
    Args:
        - name0: str
        - name1: str
        - vmax: float
        - cmap: str / MatplotlibColorbar
        - surname: str
        - masker: NiftiMasker
        - height_control: str ('fdr', 'bonferroni')
        - alpha: float
        - saving_folder: str
    """
    if surname is None:
        surname = name0 + '-vs-' + 
    # Compute difference at the group level
    zmaps_dict, effmaps_dict, masks_dict, vmax_list, fdr_th = compute_diff_group_level_maps(name0, name1)
    zmaps_dict = {surname: list(zmaps_dict.values())[0]}
    effmaps_dict = {surname: list(effmaps_dict.values())[0]}
    dict_ = {surname: {'cmap': cmap, 'vmax': vmax}}
    # If masker look at R scores distribution in the masker
    if masker is not None:
        plt.hist(masker.transform(list(effmaps_dict.values())[0]).reshape(-1), bins=200)
        plt.show()
    # Plot surface maps
    saving_folder = os.path.join(saving_folder, surname + f'_z-{height_control}_{fdr_th}')
    check_folder(saving_folder)
    plot_group_level_surface_maps(
        effmaps_dict, 
        zmaps_dict, 
        dict_, 
        threshold=1e-15, 
        height_control=height_control, 
        alpha=alpha, 
        saving_folder=saving_folder
    )
    return effmaps_dict, zmaps_dict


if __name__=='__main__':

    ###
    # Replicating Fig3 Panel A: Brain Fit
    ###
    saving_folder = 'derivatives/Fig3/panelA'
    masker = fetch_masker('derivatives/masker', None, **{'detrend': False, 'standardize': False})
    imgs = {}

    # GloVe Syntax
    name0 = "GloVe_Syntax_fullcv-alpha_-3_to_4wordrate-rms_chris-log_frequency_{}"
    name1 = "BF_wordrate-rms_chris-log_frequency_cv-alpha_-3_to_4__{}"
    effmaps_dict, zmaps_dict = plot_brain_fit(
        name0, name1, vmax=0.1, cmap='cold_hot', surname="GloVe_Syntax+BF_vs_BF", 
        masker=masker, height_control='fdr', alpha=0.005, saving_folder=saving_folder)
    imgs["GloVe_Syntax+BF_vs_BF"] = [effmaps_dict, zmaps_dict]
    
    # GloVe Semantic
    name0 = "GloVe_Semanticcv-alpha_-3_to_4wordrate-rms_chris-log_frequency_{}"
    name1 = "BF_wordrate-rms_chris-log_frequency_cv-alpha_-3_to_4__{}"
    effmaps_dict, zmaps_dict = plot_brain_fit(
        name0, name1, vmax=0.1, cmap=green, surname="GloVe_Semantic+BF_vs_BF", 
        masker=masker, height_control='fdr', alpha=0.005, saving_folder=saving_folder)
    imgs["GloVe_Semantic+BF_vs_BF"] = [effmaps_dict, zmaps_dict]
    
    # GPT-2 Syntax
    name0 = "GPT-2_L-4_H-768_Syntax_context-full_end-epoch-4_split-3_norm-None_temporal-shifting-0_{}_hidden-layer-3cv-alpha_-3_to_4_no-centering_wordrate-rms_chris-log_frequency"
    name1 = "BF_wordrate-rms_chris-log_frequency_cv-alpha_-3_to_4__{}"
    effmaps_dict, zmaps_dict = plot_brain_fit(
        name0, name1, vmax=0.1, cmap='cold_hot', surname="GPT-2_Syntax+BF_vs_BF", 
        masker=masker, height_control='fdr', alpha=0.005, saving_folder=saving_folder)
    imgs["GPT-2_Syntax+BF_vs_BF"] = [effmaps_dict, zmaps_dict]
    
    # GPT-2 Semantic
    name0 = "GloVe_Syntax_fullcv-alpha_-3_to_4wordrate-rms_chris-log_frequency_{}"
    name1 = "BF_wordrate-rms_chris-log_frequency_cv-alpha_-3_to_4__{}"
    effmaps_dict, zmaps_dict = plot_brain_fit(
        name0, name1, vmax=0.1, cmap=green, surname="GPT-2_Semantic+BF_vs_BF", 
        masker=masker, height_control='fdr', alpha=0.005, saving_folder=saving_folder)
    imgs["GPT-2_Semantic+BF_vs_BF"] = [effmaps_dict, zmaps_dict]
    
    ###
    # Replicating Fig3 Panel B: Peak regions
    ###
    saving_folder = 'derivatives/Fig3/panelB'
    ref_img = masker.mask_img_

    GPT2_syntax_zmaps_dict = imgs["GPT-2_Syntax+BF_vs_BF"][1]
    GPT2_syntax_effmaps_dict = imgs["GPT-2_Syntax+BF_vs_BF"][0]
    GloVe_syntax_zmaps_dict = imgs["GloVe_Syntax+BF_vs_BF"][1]
    GloVe_syntax_effmaps_dict = imgs["GloVe_Syntax+BF_vs_BF"][0]
    GPT2_semantic_zmaps_dict = imgs["GPT-2_Semantic+BF_vs_BF"][1]
    GPT2_semantic_effmaps_dict = imgs["GPT-2_Semantic+BF_vs_BF"][0]
    GloVe_semantic_zmaps_dict = imgs["GloVe_Semantic+BF_vs_BF"][1]
    GloVe_semantic_effmaps_dict = imgs["GloVe_Semantic+BF_vs_BF"][0]

    names_restricted = ["GloVe_Syntax+BF_vs_BF", "GloVe_Semantic+BF_vs_BF"]
    surname = 'GloVe'

    imgs = [
        GloVe_syntax_effmaps_dict["GloVe_Syntax+BF_vs_BF"],
        GloVe_semantic_effmaps_dict["GloVe_Semantic+BF_vs_BF"],
    ]
    zmaps = [
        GloVe_syntax_zmaps_dict["GloVe_Syntax+BF_vs_BF"],
        GloVe_semantic_zmaps_dict["GloVe_Semantic+BF_vs_BF"],
    ]
    plot_peak_regions(names_restricted, surname, imgs, zmaps, threshold=90, saving_folder=saving_folder, ref_img=ref_img)

    names_restricted = ["GPT-2_Syntax+BF_vs_BF", "GPT-2_Semantic+BF_vs_BF"]
    surname = 'GPT-2'

    imgs = [
        GPT2_syntax_effmaps_dict["GPT-2_Syntax+BF_vs_BF"],
        GPT2_semantic_effmaps_dict["GPT-2_Semantic+BF_vs_BF"],
    ]
    zmaps = [
        GPT2_syntax_zmaps_dict["GPT-2_Syntax+BF_vs_BF"],
        GPT2_semantic_zmaps_dict["GPT-2_Semantic+BF_vs_BF"],
    ]
    plot_peak_regions(names_restricted, surname, imgs, zmaps, threshold=90, saving_folder=saving_folder ,  ref_img=ref_img)
