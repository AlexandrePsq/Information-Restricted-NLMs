import numpy as np
from matplotlib.colors import ListedColormap

from irnlm.utils import check_folder
from fmri_encoder.data import fetch_masker
from fmri_encoder.plotting import plot_colorbar
from irnlm.contrasts import plot_brain_fit
from irnlm.peak_regions import plot_peak_regions

green = np.array([[i / 50, 1, 0, 1] for i in range(50)])
green = np.vstack(
    [
        np.array([[0, i / 100, 0, 1] for i in range(100)]),
        green,
        np.array([[1, 1, i / 25, 1] for i in range(25)]),
    ]
)
green = ListedColormap(np.vstack([green]))
plot_colorbar(green, 0.05, 0)


if __name__ == "__main__":
    ###
    # Replicating Fig3 Panel A: Brain Fit
    ###
    saving_folder = "derivatives/Fig3/panelA"
    check_folder(saving_folder)
    masker = fetch_masker(
        "derivatives/masker", None, **{"detrend": False, "standardize": False}
    )
    imgs = {}

    # GloVe Syntax
    name0 = "GloVe_Syntax_fullcv-alpha_-3_to_4wordrate-rms_chris-log_frequency_{}"
    name1 = "BF_wordrate-rms_chris-log_frequency_cv-alpha_-3_to_4__{}"
    effmaps_dict, zmaps_dict = plot_brain_fit(
        name0,
        name1,
        vmax=0.1,
        cmap="cold_hot",
        surname="GloVe_Syntax+BF_vs_BF",
        masker=masker,
        height_control="fdr",
        alpha=0.005,
        saving_folder=saving_folder,
    )
    imgs["GloVe_Syntax+BF_vs_BF"] = [effmaps_dict, zmaps_dict]

    # GloVe Semantic
    name0 = "GloVe_Semanticcv-alpha_-3_to_4wordrate-rms_chris-log_frequency_{}"
    name1 = "BF_wordrate-rms_chris-log_frequency_cv-alpha_-3_to_4__{}"
    effmaps_dict, zmaps_dict = plot_brain_fit(
        name0,
        name1,
        vmax=0.1,
        cmap=green,
        surname="GloVe_Semantic+BF_vs_BF",
        masker=masker,
        height_control="fdr",
        alpha=0.005,
        saving_folder=saving_folder,
    )
    imgs["GloVe_Semantic+BF_vs_BF"] = [effmaps_dict, zmaps_dict]

    # GPT-2 Syntax
    name0 = "GPT-2_L-4_H-768_Syntax_context-full_end-epoch-4_split-3_norm-None_temporal-shifting-0_{}_hidden-layer-3cv-alpha_-3_to_4_no-centering_wordrate-rms_chris-log_frequency"
    name1 = "BF_wordrate-rms_chris-log_frequency_cv-alpha_-3_to_4__{}"
    effmaps_dict, zmaps_dict = plot_brain_fit(
        name0,
        name1,
        vmax=0.1,
        cmap="cold_hot",
        surname="GPT-2_Syntax+BF_vs_BF",
        masker=masker,
        height_control="fdr",
        alpha=0.005,
        saving_folder=saving_folder,
    )
    imgs["GPT-2_Syntax+BF_vs_BF"] = [effmaps_dict, zmaps_dict]

    # GPT-2 Semantic
    name0 = "GloVe_Syntax_fullcv-alpha_-3_to_4wordrate-rms_chris-log_frequency_{}"
    name1 = "BF_wordrate-rms_chris-log_frequency_cv-alpha_-3_to_4__{}"
    effmaps_dict, zmaps_dict = plot_brain_fit(
        name0,
        name1,
        vmax=0.1,
        cmap=green,
        surname="GPT-2_Semantic+BF_vs_BF",
        masker=masker,
        height_control="fdr",
        alpha=0.005,
        saving_folder=saving_folder,
    )
    imgs["GPT-2_Semantic+BF_vs_BF"] = [effmaps_dict, zmaps_dict]

    ###
    # Replicating Fig3 Panel B: Peak regions
    ###
    saving_folder = "derivatives/Fig3/panelB"
    ref_img = masker.mask_img_

    GPT2_syntax_zmaps = imgs["GPT-2_Syntax+BF_vs_BF"][1]
    GPT2_syntax_effmaps = imgs["GPT-2_Syntax+BF_vs_BF"][0]
    GloVe_syntax_zmaps = imgs["GloVe_Syntax+BF_vs_BF"][1]
    GloVe_syntax_effmaps = imgs["GloVe_Syntax+BF_vs_BF"][0]
    GPT2_semantic_zmaps = imgs["GPT-2_Semantic+BF_vs_BF"][1]
    GPT2_semantic_effmaps = imgs["GPT-2_Semantic+BF_vs_BF"][0]
    GloVe_semantic_zmaps = imgs["GloVe_Semantic+BF_vs_BF"][1]
    GloVe_semantic_effmaps = imgs["GloVe_Semantic+BF_vs_BF"][0]

    names_restricted = ["GloVe_Syntax+BF_vs_BF", "GloVe_Semantic+BF_vs_BF"]
    surname = "GloVe"
    imgs = [GloVe_syntax_effmaps, GloVe_semantic_effmaps]
    zmaps = [GloVe_syntax_zmaps, GloVe_semantic_zmaps]
    plot_peak_regions(
        names_restricted,
        surname,
        imgs,
        zmaps,
        masker,
        threshold=90,
        saving_folder=saving_folder,
        ref_img=ref_img,
    )

    names_restricted = ["GPT-2_Syntax+BF_vs_BF", "GPT-2_Semantic+BF_vs_BF"]
    surname = "GPT-2"
    imgs = [GPT2_syntax_effmaps, GPT2_semantic_effmaps]
    zmaps = [GPT2_syntax_zmaps, GPT2_semantic_zmaps]
    plot_peak_regions(
        names_restricted,
        surname,
        imgs,
        zmaps,
        masker,
        threshold=90,
        saving_folder=saving_folder,
        ref_img=ref_img,
    )
