# %%
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import os, glob
import numpy as np
import nibabel as nib
from matplotlib.colors import ListedColormap
from nilearn.image import math_img
from irnlm.utils import check_folder
from fmri_encoder.data import fetch_masker
from fmri_encoder.plotting import plot_colorbar
from irnlm.contrasts import compute_diff_imgs, plot_group_level_surface_maps
from irnlm.maskers import fetch_atlas_labels, get_roi_mask
from irnlm.peak_regions import subj_peak_regions
from irnlm.utils import get_progress, save_yaml, rich_progress_joblib

from joblib import delayed, Parallel

import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


green = np.array([[i / 50, 1, 0, 1] for i in range(50)])
green = np.vstack(
    [
        np.array([[0, i / 100, 0, 1] for i in range(100)]),
        green,
        np.array([[1, 1, i / 25, 1] for i in range(25)]),
    ]
)

green = ListedColormap(np.vstack([green]))
# plot_colorbar(green, 0.05, 0)
yellow = np.array([[i / 200, i / 200, 0, 1] for i in range(200)])
yellow = np.vstack([yellow, np.array([[1, 1, i / 50, 1] for i in range(50)])])
yellow = ListedColormap(np.vstack([yellow]))


# %%
def get_subject_name(id):
    """Get subject name from id.
    Arguments:
        - id: int
    Returns:
        - str
    """
    if type(id) == str:
        return "sub-{}".format(id)
    elif id < 10:
        return "sub-00{}".format(id)
    elif id < 100:
        return "sub-0{}".format(id)
    else:
        return "sub-{}".format(id)


def possible_subjects_id(language):
    """Returns possible subject id list for a given language.
    Arguments:
        - language: str
    Returns:
        result: list (of int)
    """
    if language == "english":
        result = [
            57,
            58,
            59,
            61,
            62,
            63,
            64,
            65,
            66,
            67,
            68,
            69,
            70,
            72,
            73,
            74,
            75,
            76,
            77,
            78,
            79,
            80,
            81,
            82,
            83,
            84,
            86,
            87,
            88,
            89,
            91,
            92,
            93,
            94,
            95,
            96,
            97,
            98,
            99,
            100,
            101,
            103,
            104,
            105,
            106,
            108,
            109,
            110,
            113,
            114,
            115,
        ]
    elif language == "french":
        result = [
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            10,
            11,
            12,
            13,
            14,
            15,
            16,
            17,
            18,
            19,
            20,
            22,
            23,
            24,
            25,
            26,
            27,
            29,
            30,
        ]  # ,54,55,56,57,58,59,61,62,63,64,65 # TO DO #21 was removed issue to investigate
    elif language == "chineese":
        result = [1]  # TO DO
    elif language == "ibc":
        result = [54, 55, 56, 57, 58, 59, 61, 62, 63, 64, 65]
    else:
        raise Exception("Language {} not known.".format(language))
    return result


subjects = [get_subject_name(id_) for id_ in possible_subjects_id("english")]
subjects_data = {
    subj: {
        "GloVe_Syntax": nib.load(
            glob.glob(
                f"/Users/alexpsq/Code/Parietal/data/stimuli-representations/english/maps/{subj}*GloVe_Syntax_fullcv-alpha_-3_to_4wordrate-r*"
            )[0]
        ),
        "GloVe_Semantic": nib.load(
            glob.glob(
                f"/Users/alexpsq/Code/Parietal/data/stimuli-representations/english/maps/{subj}*GloVe_Semanticcv-alpha_-3_*"
            )[0]
        ),
        "GloVe": nib.load(
            glob.glob(
                f"/Users/alexpsq/Code/Parietal/data/stimuli-representations/english/maps/{subj}*GloVecv-alpha_-3_to_4wo*"
            )[0]
        ),
        "GPT-2_Syntax": nib.load(
            glob.glob(
                f"/Users/alexpsq/Code/Parietal/data/stimuli-representations/english/maps/{subj}*GPT-2_L-4_H-768_Syntax_context-full_end-epoch-4*"
            )[0]
        ),
        "GPT-2_Semantic": nib.load(
            glob.glob(
                f"/Users/alexpsq/Code/Parietal/data/stimuli-representations/english/maps/{subj}*GPT-2_L-4_H-768_semantic_context-full_end-epoch-4_spl*"
            )[0]
        ),
        "GPT-2": nib.load(
            glob.glob(
                f"/Users/alexpsq/Code/Parietal/data/stimuli-representations/english/maps/{subj}*GPT-2_L-4_H-768_default_tokenizer_full_end-epoch-4_*"
            )[0]
        ),
        "BF": nib.load(
            glob.glob(
                f"/Users/alexpsq/Code/Parietal/data/stimuli-representations/english/maps/{subj}*BF_wordrate*"
            )[0]
        ),
    }
    for subj in subjects
}


# %%
atlas_maps, labels = fetch_atlas_labels("harvard-oxford")
# %%
from irnlm.maskers import create_mask_from_threshold

results = {}
saving_folder = "derivatives/Fig3/panelA"
check_folder(saving_folder)
masker = fetch_masker(
    "/Users/alexpsq/Code/Parietal/LePetitPrince/derivatives/fMRI/ROI_masks/global_masker_english",
    None,
    **{"detrend": False, "standardize": False},
)


def process(atlas_maps, index_roi, labels, subjects, subjects_data):
    test_img = img_glove_syntax = subjects_data[subjects[0]]["GloVe_Syntax"]
    masker_roi = get_roi_mask(
        atlas_maps,
        index_roi,
        labels,
        path=None,
        global_mask=None,
        resample_to_img_=test_img,
        intersect_with_img=False,
        PROJECT_PATH="/Users/alexpsq/Code/Parietal/LePetitPrince/",
    )
    results = {}
    results["glove_syntax"] = []
    results["glove_semantic"] = []
    results["gpt2_syntax"] = []
    results["gpt2_semantic"] = []
    for subj in subjects:
        img_glove_syntax = subjects_data[subj]["GloVe_Syntax"]
        img_glove_semantic = subjects_data[subj]["GloVe_Semantic"]
        # img_glove = subjects_data[subj]["GloVe"]
        img_gpt2_syntax = subjects_data[subj]["GPT-2_Syntax"]
        img_gpt2_semantic = subjects_data[subj]["GPT-2_Semantic"]
        # img_gpt2 = subjects_data[subj]["GPT-2"]
        img_BF = subjects_data[subj]["BF"]

        syntax_GloVe = {
            f"{subj}_GloVe_Syntax+BF_vs_BF": math_img(
                "img1-img2", img1=img_glove_syntax, img2=img_BF
            )
        }
        semantic_GloVe = {
            f"{subj}_GloVe_Semantic+BF_vs_BF": math_img(
                "img1-img2", img1=img_glove_semantic, img2=img_BF
            )
        }
        syntax_GPT2 = {
            f"{subj}_GPT-2_Syntax+BF_vs_BF": math_img(
                "img1-img2", img1=img_gpt2_syntax, img2=img_BF
            )
        }
        semantic_GPT2 = {
            f"{subj}_GPT-2_Semantic+BF_vs_BF": math_img(
                "img1-img2", img1=img_gpt2_semantic, img2=img_BF
            )
        }
        # dict_ = {f"{subj}_GloVe_Syntax+BF_vs_BF": {"cmap": "cold_hot", "vmax": 0.35}}

        imgs_glove = [
            syntax_GloVe[f"{subj}_GloVe_Syntax+BF_vs_BF"],
            semantic_GloVe[f"{subj}_GloVe_Semantic+BF_vs_BF"],
        ]
        imgs_gpt2 = [
            syntax_GPT2[f"{subj}_GPT-2_Syntax+BF_vs_BF"],
            semantic_GPT2[f"{subj}_GPT-2_Semantic+BF_vs_BF"],
        ]
        overlap_masks_glove = [
            create_mask_from_threshold(
                img, masker, threshold=90, above=True, percentile=True
            )
            for img in imgs_glove
        ]
        results["glove_syntax"].append(masker_roi.transform(overlap_masks_glove[0]))
        results["glove_semantic"].append(masker_roi.transform(overlap_masks_glove[1]))

        overlap_masks_gpt2 = [
            create_mask_from_threshold(
                img, masker, threshold=90, above=True, percentile=True
            )
            for img in imgs_gpt2
        ]
        results["gpt2_syntax"].append(masker_roi.transform(overlap_masks_gpt2[0]))
        results["gpt2_semantic"].append(masker_roi.transform(overlap_masks_gpt2[1]))
    return results


with rich_progress_joblib(
    "Computing",
    total=len(labels),
    verbose=True,
):
    results = Parallel(n_jobs=-1)(
        delayed(process)(atlas_maps, index_roi, labels, subjects, subjects_data)
        for index_roi in range(len(labels))
    )

# %%
params = masker.get_params()
from nilearn import plotting


def multi_masker(
    atlas_maps,
    index_list_mask,
    labels,
    params,
    resample_to_img_=None,
    intersect_with_img=False,
    PROJECT_PATH="/Users/alexpsq/Code/Parietal/LePetitPrince/",
):
    """Create a mask from the superposition of several sub-masks (could be voxel masks or ROI masks).
    Returns the list of superposed masks and the final mask image.
    Args:
        - atlas_maps: NiftiImage (atlas image)
        - index_list_mask: list of int (each int indexes an ROI from labels)
        - labels: list of ROI names
        - resample_to_img: NiftiImage (or None)
        - intersect_with_img: bool
        - PROJECT_PATH: string
    Returns:
        - masks: list of NiftiImage
        - result_img: NiftiImage
    """
    masks = []
    for index_mask in index_list_mask:
        masks.append(
            get_roi_mask(
                atlas_maps,
                index_mask,
                labels,
                resample_to_img_=resample_to_img_,
                intersect_with_img=intersect_with_img,
                PROJECT_PATH=PROJECT_PATH,
            )
        )
    result_img = masks[0].mask_img
    for mask in masks[1:]:
        result_img = math_img("img1 + img2", img1=result_img, img2=mask.mask_img)

    from nilearn.maskers import NiftiMasker

    masker = NiftiMasker(result_img)
    masker.set_params(**params)
    masker.fit()
    return masker, result_img


# %%
colors = {
    "Left Superior Frontal Gyrus": "red",
    "Right Superior Frontal Gyrus": "blue",
    "Left Middle Frontal Gyrus": "green",
    "Right Middle Frontal Gyrus": "green",
    "Left Inferior Frontal Gyrus, pars triangularis": "green",
    "Right Inferior Frontal Gyrus, pars triangularis": "blue",
    "Left Inferior Frontal Gyrus, pars opercularis": "red",
    "Right Inferior Frontal Gyrus, pars opercularis": "blue",
    "Right Temporal Pole": "red",
    "Left Superior Temporal Gyrus, anterior division": "red",
    "Right Superior Temporal Gyrus, anterior division": "red",
    "Left Superior Temporal Gyrus, posterior division": "red",
    "Right Superior Temporal Gyrus, posterior division": "red",
    "Right Middle Temporal Gyrus, anterior division": "red",
    "Right Middle Temporal Gyrus, posterior division": "red",
    "Left Middle Temporal Gyrus, temporooccipital part": "green",
    "Right Middle Temporal Gyrus, temporooccipital part": "blue",
    "Right Inferior Temporal Gyrus, anterior division": "red",
    "Left Inferior Temporal Gyrus, temporooccipital part": "green",
    "Left Supramarginal Gyrus, anterior division": "green",
    "Left Supramarginal Gyrus, posterior division": "blue",
    "Right Supramarginal Gyrus, posterior division": "blue",
    "Left Angular Gyrus": "blue",
    "Right Angular Gyrus": "blue",
    "Left Lateral Occipital Cortex, superior division": "green",
    "Right Lateral Occipital Cortex, superior division": "green",
    "Left Cingulate Gyrus, posterior division": "green",
    "Right Cingulate Gyrus, posterior division": "green",
    "Left Precuneous Cortex": "green",
    "Right Precuneous Cortex": "green",
}
# %%
import os
import numpy as np

from nilearn.image import math_img
from nilearn import plotting

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from fmri_encoder.plotting import (
    concat_colormaps,
    plot_colorbar,
    create_grid,
    set_projection_params,
    compute_surf_proj,
)
from irnlm.contrasts import mask_from_zmap
from irnlm.maskers import intersect_binary, create_mask_from_threshold, masks_overlap

categorical_values = True
inflated = False
hemispheres = ["left", "right"]
views = ["lateral", "medial"]

for key in ["glove_syntax", "glove_semantic", "gpt2_syntax", "gpt2_semantic"]:
    print(key)
    roi_size = [np.vstack(roi[key]).shape[-1] for i, roi in enumerate(results)]
    # roi_res = [
    #    np.sum(np.sum(np.vstack(roi[key]), axis=-1) > roi_size[i] / 10)
    #    for i, roi in enumerate(results)
    # ]
    roi_res = [
        np.mean(np.sum(np.vstack(roi[key]), axis=-1) / roi_size[i] * 100)
        for i, roi in enumerate(results)
    ]
    error = [
        np.std(np.sum(np.vstack(roi[key]), axis=-1) / roi_size[i] * 100)
        for i, roi in enumerate(results)
    ]
    # roi_res = [np.vstack(roi[key]).shape for roi in results]
    indexes = np.argsort(roi_res)[::-1]
    indexes = [ind for ind in indexes if labels[ind] in colors.keys()]
    # indexes = [ind for ind in indexes if roi_res[ind] > 39]
    rois = [labels[i] for i in indexes]
    # print(roi_res)
    plt.figure(figsize=(5, 15))
    ax = plt.subplot(111)
    ax.barh(
        np.arange(len(indexes)),
        [roi_res[i] for i in indexes],
        xerr=[error[i] / np.sqrt(51) for i in indexes],
        color=[colors[roi] for roi in rois],
    )
    ax.set_yticks(np.arange(len(indexes)))
    ax.set_yticklabels(rois, rotation=0, fontsize=15)
    ax.set_xticks([0, 10, 20, 30])
    ax.set_xlim((0, 30))
    ax.set_xticklabels([0, 10, 20, 30], rotation=0, fontsize=15)
    # plt.xticks(np.arange(len(indexes)), rois, rotation=90, fontsize=10)
    for i, xtick in enumerate(ax.get_yticklabels()):
        color = colors[rois[i]]
        xtick.set_color(color)
    ax.invert_yaxis()
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    from matplotlib.lines import Line2D

    custom_lines = [
        Line2D([0], [0], color="red", lw=4),
        Line2D([0], [0], color="green", lw=4),
        Line2D([0], [0], color="blue", lw=4),
    ]

    ax.legend(
        custom_lines, ["Syntax", "Semantics", "Both"], loc="lower right", fontsize=15
    )
    ax.set_title(key.replace("_", " "), fontsize=30)
    plt.savefig(
        os.path.join(saving_folder, f"{key}.pdf"),
        format="pdf",
        dpi=300,
        bbox_inches="tight",
        pad_inches=0,
    )

    plt.show()
    # masker_key, result_img = multi_masker(atlas_maps, indexes, labels, params)
    # plotting.plot_glass_brain(result_img, display_mode="lzry")
    # surf_projs_syn_sem_core_superimposed = compute_surf_proj(
    #    [result_img],
    #    zmaps=None,
    #    masks=None,
    #    ref_img=masker.mask_img_,
    #    names=[key],
    #    categorical_values=None,
    #    inflated=False,
    #    hemispheres=["left", "right"],
    #    views=["lateral", "medial"],
    #    kind="line",
    #    template=None,
    # )
#
# values = {"left": None, "right": None}
# distributions = {
#    "left": {"lateral": None, "medial": None},
#    "right": {"lateral": None, "medial": None},
# }
# for h, hemi in enumerate(hemispheres):
#    for v, view in enumerate(views):
#        figure, axes = create_grid(
#            nb_rows=1,
#            nb_columns=1,
#            row_size_factor=6,
#            overlapping=6,
#            column_size_factor=12,
#        )
#
#        # if hemi=='left' and view=='lateral':
#        ax = axes[0][0]
#        kwargs = set_projection_params(
#            hemi,
#            view,
#            cmap="Greys",
#            inflated=False,
#            threshold=1e-15,
#            colorbar=False,
#            symmetric_cbar=False,
#            template=None,
#            figure=figure,
#            ax=ax,
#            vmax=1.1,
#        )
#
#        surf_imgs = [
#            surf_projs_syn_sem_core_superimposed[name][f"{hemi}-{view}"]
#            for name in [key]
#        ]
#        d = [
#            np.hstack(
#                [
#                    surf_projs_syn_sem_core_superimposed[name][f"{hemi}-lateral"],
#                    surf_projs_syn_sem_core_superimposed[name][f"{hemi}-medial"],
#                ]
#            )
#            for name in [key]
#        ]
#
#        for k, surf_img in enumerate(surf_imgs):
#            surf_imgs[k][np.isnan(surf_img)] = 0
#
#        if categorical_values:
#            plotting.plot_surf_roi(roi_map=surf_img, **kwargs)
#        else:
#            plotting.plot_surf_stat_map(stat_map=surf_img, **kwargs)
#        plot_name = f"{key}_{hemi}_{view}_roi_analysis"
#        format_figure = "pdf"
#        dpi = 300
#        plt.savefig(
#            os.path.join(saving_folder, f"{plot_name}.{format_figure}"),
#            format=format_figure,
#            dpi=dpi,
#            bbox_inches="tight",
#            pad_inches=0,
#        )
#        # plt.show()
#        plt.close("all")

# %%


# %%

if __name__ == "__main__":
    ###
    # Replicating Fig3 Panel A: Brain Fit
    ###
    saving_folder = "derivatives/Fig3/panelA"
    check_folder(saving_folder)
    masker = fetch_masker(
        "/Users/alexpsq/Code/Parietal/LePetitPrince/derivatives/fMRI/ROI_masks/global_masker_english",
        None,
        **{"detrend": False, "standardize": False},
    )
    imgs = {}
    with get_progress(transient=True) as progress:
        task = progress.add_task(f"Computing", total=len(subjects))
        for subj in subjects:
            img_glove_syntax = subjects_data[subj]["GloVe_Syntax"]
            img_glove_semantic = subjects_data[subj]["GloVe_Semantic"]
            img_glove = subjects_data[subj]["GloVe"]
            img_gpt2_syntax = subjects_data[subj]["GPT-2_Syntax"]
            img_gpt2_semantic = subjects_data[subj]["GPT-2_Semantic"]
            img_gpt2 = subjects_data[subj]["GPT-2"]
            img_BF = subjects_data[subj]["BF"]

            syntax_GloVe = {
                f"{subj}_GloVe_Syntax+BF_vs_BF": math_img(
                    "img1-img2", img1=img_glove_syntax, img2=img_BF
                )
            }
            semantic_GloVe = {
                f"{subj}_GloVe_Semantic+BF_vs_BF": math_img(
                    "img1-img2", img1=img_glove_semantic, img2=img_BF
                )
            }
            syntax_GPT2 = {
                f"{subj}_GPT-2_Syntax+BF_vs_BF": math_img(
                    "img1-img2", img1=img_gpt2_syntax, img2=img_BF
                )
            }
            semantic_GPT2 = {
                f"{subj}_GPT-2_Semantic+BF_vs_BF": math_img(
                    "img1-img2", img1=img_gpt2_semantic, img2=img_BF
                )
            }
            dict_ = {
                f"{subj}_GloVe_Syntax+BF_vs_BF": {"cmap": "cold_hot", "vmax": 0.35}
            }
            plot_group_level_surface_maps(
                syntax_GloVe,
                None,
                dict_,
                ref_img=masker.mask_img_,
                threshold=1e-15,
                height_control="fdr",
                alpha=0.005,
                saving_folder=saving_folder,
            )
            dict_ = {f"{subj}_GloVe_Semantic+BF_vs_BF": {"cmap": green, "vmax": 0.35}}
            plot_group_level_surface_maps(
                semantic_GloVe,
                None,
                dict_,
                ref_img=masker.mask_img_,
                threshold=1e-15,
                height_control="fdr",
                alpha=0.005,
                saving_folder=saving_folder,
            )
            dict_ = {
                f"{subj}_GPT-2_Syntax+BF_vs_BF": {"cmap": "cold_hot", "vmax": 0.35}
            }
            plot_group_level_surface_maps(
                syntax_GPT2,
                None,
                dict_,
                ref_img=masker.mask_img_,
                threshold=1e-15,
                height_control="fdr",
                alpha=0.005,
                saving_folder=saving_folder,
            )
            dict_ = {f"{subj}_GPT-2_Semantic+BF_vs_BF": {"cmap": green, "vmax": 0.35}}
            plot_group_level_surface_maps(
                semantic_GPT2,
                None,
                dict_,
                ref_img=masker.mask_img_,
                threshold=1e-15,
                height_control="fdr",
                alpha=0.005,
                saving_folder=saving_folder,
            )

            ###
            # Replicating Fig3 Panel B: Peak regions
            ###
            saving_folder2 = "derivatives/Fig3/panelB"
            ref_img = masker.mask_img_

            names_restricted = [
                f"{subj}_GloVe_Syntax+BF_vs_BF",
                f"{subj}_GloVe_Semantic+BF_vs_BF",
            ]
            surname = "GloVe"
            imgs = [
                syntax_GloVe[f"{subj}_GloVe_Syntax+BF_vs_BF"],
                semantic_GloVe[f"{subj}_GloVe_Semantic+BF_vs_BF"],
            ]
            values, distributions = subj_peak_regions(
                names_restricted,
                surname,
                imgs,
                masker,
                subj,
                threshold=90,
                saving_folder=saving_folder2,
                ref_img=ref_img,
            )
            save_yaml(values, os.path.join(saving_folder, f"{subj}_GloVe_values.yml"))
            save_yaml(
                distributions,
                os.path.join(saving_folder, f"{subj}_GloVe_distributions.yml"),
            )

            names_restricted = [
                f"{subj}_GPT-2_Syntax+BF_vs_BF",
                f"{subj}_GPT-2_Semantic+BF_vs_BF",
            ]
            surname = "GPT-2"
            imgs = [
                syntax_GPT2[f"{subj}_GPT-2_Syntax+BF_vs_BF"],
                semantic_GPT2[f"{subj}_GPT-2_Semantic+BF_vs_BF"],
            ]
            values, distributions = subj_peak_regions(
                names_restricted,
                surname,
                imgs,
                masker,
                subj,
                threshold=90,
                saving_folder=saving_folder2,
                ref_img=ref_img,
            )
            save_yaml(values, os.path.join(saving_folder, f"{subj}_GPT-2_values.yml"))
            save_yaml(
                distributions,
                os.path.join(saving_folder, f"{subj}_GPT-2_distributions.yml"),
            )
            progress.update(task, advance=1)
