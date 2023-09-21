import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.stats import norm

import nibabel as nib
from nilearn.glm import threshold_stats_img
from nilearn.plotting import plot_surf_stat_map
from nilearn.glm.second_level import SecondLevelModel
from nilearn.image import math_img, new_img_like

import matplotlib.pyplot as plt

from fmri_encoder.plotting import compute_surf_proj, create_grid, set_projection_params
from irnlm.utils import check_folder
from irnlm.data.utils import fetch_model_maps


def mask_from_zmap(zmap, ref_img):
    """Get the binary mask from zmap."""
    thresholded_zmap, fdr_th = threshold_stats_img(
        stat_img=math_img("img1*img2", img1=zmap, img2=ref_img),
        alpha=0.1,
        height_control="bonferroni",
        cluster_threshold=30,
        mask_img=ref_img,
    )
    mask_bonf = new_img_like(zmap, (np.abs(thresholded_zmap.get_fdata()) > 0))
    mask = math_img("img1*img2", img1=mask_bonf, img2=ref_img)
    return mask, fdr_th


def group_analysis(
    imgs,
    masker,
    smoothing_fwhm=6,
    vmax=None,
    design_matrix=None,
    p_val=None,
    fdr=0.005,
    mask_img=None,
    cluster_threshold=30,
    height_control="fdr",
):
    """Compute the difference between two images.
    Args:
        - imgs: list of NifitImages
        - smoothing_fwhm: int
        - vmax: float
        - design_matrix: pandas dataframe / None
        - p_val: float
        - fdr: float
        - mask_img: Niimg-like, NiftiMasker or MultiNiftiMasker object, optional
        - cluster_threshold: int
        - height_control: str
    Returns:
        - zmap: Nifti image
        - eff_map: Nifit image
        - mask: NifitMasker, mask with FDR/Bonferroni correction
        - fdr_th: float, threshold set by max(FDR correction, input p-value)
    """
    # Creating Second Level Model and fitting it
    plt.hist(masker.transform(imgs).reshape(-1), bins=200)
    plt.show()
    plt.hist(
        np.mean(np.stack(masker.transform(imgs), axis=0), axis=0).reshape(-1), bins=200
    )
    plt.show()
    model = SecondLevelModel(
        smoothing_fwhm=smoothing_fwhm, n_jobs=-1, mask_img=mask_img
    )
    design_matrix = (
        design_matrix
        if (design_matrix is not None)
        else pd.DataFrame([1] * len(imgs), columns=["intercept"])
    )
    model = model.fit(imgs, design_matrix=design_matrix)
    # Computing Z/effect-size maps and z-threshold
    z_map = model.compute_contrast(output_type="z_score")
    eff_map = model.compute_contrast(output_type="effect_size")

    # Applying FDR/Bonferroni correction to zmap
    mask_img = (
        mask_img
        if type(mask_img) == nib.nifti1.Nifti1Image
        else (None if mask_img is None else mask_img.mask_img_)
    )
    thresholded_zmap, fdr_th = threshold_stats_img(
        stat_img=z_map,
        alpha=fdr,
        height_control=height_control,
        cluster_threshold=cluster_threshold,
        mask_img=mask_img,
    )
    mask = new_img_like(eff_map, (np.abs(thresholded_zmap.get_fdata()) > 0))

    # Additional thresholding from input p-value
    if p_val is not None:
        z_th = norm.isf(p_val)
        mask = new_img_like(eff_map, (np.abs(thresholded_zmap.get_fdata()) > z_th))
        fdr_th = max(fdr_th, z_th)

    return z_map, eff_map, mask, fdr_th


def compute_diff_imgs(img1s, img2s):
    """Compute the difference between two list of images element-wised.
    Args:
        - img1: list of NifitImages
        - img2: list of NifitImages
    Returns:
        - diff_imgs: list of NifitImages
    """
    diff_imgs = [
        math_img("img1-img2", img1=img1, img2=img2)
        for (img1, img2) in zip(img1s, img2s)
    ]
    return diff_imgs


def compute_group_level_maps(names, masker):
    """ """
    data = fetch_model_maps(
        names,
        language="english",
        verbose=0,
        FMRIDATA_PATH="/neurospin/unicog/protocols/IRMf/LePetitPrince_Pallier_2018/LePetitPrince/derivatives/fMRI/maps/english",
    )
    imgs = {key: data[key]["Pearson_coeff"] for key in data.keys()}
    zmaps_dict = {}
    effmaps_dict = {}
    masks_dict = {}
    for key in tqdm(imgs.keys()):
        zmap, effmap, mask, fdr_th = group_analysis(
            imgs[key],
            smoothing_fwhm=6,
            vmax=None,
            design_matrix=None,
            p_val=None,
            fdr=0.005,
            mask_img=masker.mask_img_,
            cluster_threshold=30,
            height_control="fdr",
        )
        zmaps_dict[key] = zmap
        effmaps_dict[key] = effmap
        masks_dict[key] = mask

    vmax_list = []
    for key in effmaps_dict.keys():
        vmax_list.append(np.max(effmaps_dict[key].get_fdata()))
    return zmaps_dict, effmaps_dict, masks_dict, vmax_list, fdr_th


def compute_diff_group_level_maps(
    name1, name2, name3=None, masker=None, verbose=0, height_control="fdr", fdr=0.005
):
    """ """
    names = [name1, name2, name3] if name3 is not None else [name1, name2]
    data = fetch_model_maps(
        names,
        language="english",
        verbose=verbose,
        FMRIDATA_PATH="/neurospin/unicog/protocols/IRMf/LePetitPrince_Pallier_2018/LePetitPrince/derivatives/fMRI/maps/english",
    )
    imgs = [data[key]["Pearson_coeff"] for key in data.keys()]
    if name3 is not None:
        imgs = {
            "-vs-".join(list(data.keys())): [
                math_img("img1-img2+img3", img1=img1, img2=img2, img3=img3)
                for (img1, img2, img3) in zip(*imgs)
            ]
        }
    else:
        imgs = {
            "-vs-".join(list(data.keys())): [
                math_img("img1-img2", img1=img1, img2=img2)
                for (img1, img2) in zip(*imgs)
            ]
        }
    zmaps_dict = {}
    effmaps_dict = {}
    masks_dict = {}
    for key in tqdm(imgs.keys()):
        zmap, effmap, mask, fdr_th = group_analysis(
            imgs[key],
            smoothing_fwhm=6,
            vmax=None,
            design_matrix=None,
            p_val=None,
            fdr=fdr,
            mask_img=masker.mask_img_,
            cluster_threshold=30,
            height_control=height_control,
        )
        zmaps_dict[key] = zmap
        effmaps_dict[key] = effmap
        masks_dict[key] = mask

    vmax_list = []
    for key in effmaps_dict.keys():
        vmax_list.append(np.max(effmaps_dict[key].get_fdata()))
    return zmaps_dict, effmaps_dict, masks_dict, vmax_list, fdr_th


def plot_group_level_surface_maps(
    imgs,
    zmaps,
    dict_,
    ref_img,
    threshold=1e-15,
    height_control="fdr",
    alpha=0.005,
    plot_cbar=False,
    format_figure="pdf",
    dpi=300,
    language="english",
    saving_folder="/neurospin/unicog/protocols/IRMf/LePetitPrince_Pallier_2018/LePetitPrince/derivatives/fMRI/38_thesis",
):
    """ """
    surnames = list(dict_.keys())
    corr_part_imgs_ = [imgs[k] for k in imgs.keys()]
    corr_part_zmaps_ = [zmaps[k] for k in zmaps.keys()] if zmaps is not None else None

    surf_projs_raw_superimposed = compute_surf_proj(
        corr_part_imgs_,
        zmaps=corr_part_zmaps_,
        masks=None,
        names=surnames,
        ref_img=ref_img,
        categorical_values=None,
        inflated=False,
        hemispheres=["left", "right"],
        views=["lateral", "medial"],
        kind="line",
        template=None,
        language=language,
    )

    hemispheres = ["left", "right"]
    views = ["lateral", "medial"]
    categorical_values = False
    inflated = False

    for i, name in enumerate(surnames):
        vmax = dict_[name]["vmax"]
        cmap = dict_[name]["cmap"]
        for h, hemi in enumerate(hemispheres):
            for v, view in enumerate(views):
                figure, axes = create_grid(
                    nb_rows=1,
                    nb_columns=1,
                    row_size_factor=6,
                    overlapping=6,
                    column_size_factor=12,
                )

                # if hemi=='left' and view=='lateral':
                ax = axes[0][0]
                kwargs = set_projection_params(
                    hemi,
                    view,
                    cmap=cmap,
                    inflated=inflated,
                    threshold=threshold,
                    colorbar=plot_cbar,
                    symmetric_cbar=False,
                    template=None,
                    figure=figure,
                    ax=ax,
                    vmax=vmax,
                )

                surf_img = surf_projs_raw_superimposed[name][f"{hemi}-{view}"]
                plot_surf_stat_map(stat_map=surf_img, **kwargs)

                plot_name = f"{name}_{hemi}_{view}"
                format_figure = format_figure
                dpi = dpi
                plt.savefig(
                    os.path.join(saving_folder, f"{plot_name}.{format_figure}"),
                    format=format_figure,
                    dpi=dpi,
                    bbox_inches="tight",
                    pad_inches=0,
                )
                # plt.show()
                plt.close("all")


def plot_brain_fit(
    name0,
    name1,
    vmax=0.1,
    cmap="cold_hot",
    surname=None,
    masker=None,
    height_control="fdr",
    alpha=0.005,
    saving_folder=None,
    show_significant=True,
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
        - show_significant: bool
    """
    if surname is None:
        surname = name0 + "-vs-" + name1
    # Compute difference at the group level
    (
        zmaps_dict,
        effmaps_dict,
        masks_dict,
        vmax_list,
        fdr_th,
    ) = compute_diff_group_level_maps(name0, name1)
    zmaps_dict = {surname: list(zmaps_dict.values())[0]}
    effmaps_dict = {surname: list(effmaps_dict.values())[0]}
    dict_ = {surname: {"cmap": cmap, "vmax": vmax}}
    # If masker look at R scores distribution in the masker
    if masker is not None:
        plt.hist(masker.transform(list(effmaps_dict.values())[0]).reshape(-1), bins=200)
        plt.show()
    # Plot surface maps
    saving_folder = os.path.join(
        saving_folder, surname + f"_z-{height_control}_{fdr_th}"
    )
    check_folder(saving_folder)
    plot_group_level_surface_maps(
        effmaps_dict,
        zmaps_dict,
        dict_,
        ref_img=masker.mask_img_,
        threshold=1e-15,
        height_control=height_control,
        alpha=alpha,
        saving_folder=saving_folder,
    )
    return effmaps_dict, zmaps_dict
