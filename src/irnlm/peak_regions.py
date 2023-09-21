import os
import numpy as np
import itertools
from sklearn.metrics import jaccard_score

from nilearn.image import math_img
from nilearn import plotting

import matplotlib.pyplot as plt
import seaborn as sns
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

yellow = np.array([[i / 200, i / 200, 0, 1] for i in range(200)])
yellow = np.vstack([yellow, np.array([[1, 1, i / 50, 1] for i in range(50)])])
yellow = ListedColormap(np.vstack([yellow]))


def plot_distrib(
    img,
    masker,
    syntax=True,
    plot_name="test",
    saving_folder="./",
    fdr_th=None,
    format_figure="pdf",
    dpi=300,
):
    """Plot R scores distribution."""
    fig = plt.figure(figsize=(10, 5))
    data = masker.transform(img).reshape(-1)

    threshold = np.percentile(data, 90)

    color = "red" if syntax else "green"
    label = (
        "Voxels in Syntactic \npeak regions"
        if syntax
        else "Voxels in Semantic \npeak regions"
    )

    # plt.fill_between(x, y2, step="pre", alpha=0.4)

    ax = sns.distplot(
        data,
        hist=False,
        kde=True,
        kde_kws={"shade": False, "linewidth": 3, "color": "blue"},
    )
    kde_x, kde_y = ax.lines[0].get_data()

    ax.fill_between(
        kde_x,
        kde_y,
        where=kde_x > threshold,  # where=(kde_x<x0) | (kde_x>threshold)
        interpolate=True,
        color=color,
        label=label,
    )
    plt.axvline(
        threshold,
        color="black",
        label=f"90-th percentile: {np.round(threshold, 3)}",
        linewidth=3,
    )
    if fdr_th is not None:
        plt.axvline(
            fdr_th,
            color="green",
            label=f"Significance: {np.round(fdr_th, 3)}",
            linewidth=3,
        )

    ax = plt.axes()
    plt.minorticks_on()
    ax.tick_params(axis="x", labelsize=30, rotation=0)
    ax.tick_params(axis="y", labelsize=30)

    # Parametrization of the figure
    plt.ylabel("Density", fontsize=35)
    plt.xlabel("R scores", fontsize=35)
    plt.ylim((0, 60))
    plt.xlim((-0.01, 0.25))
    plt.xticks(np.arange(0, 21, 5) / 100, np.arange(0, 21, 5) / 100)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.legend(fontsize=30)
    plt.savefig(
        os.path.join(saving_folder, f"{plot_name}.{format_figure}"),
        format=format_figure,
        dpi=dpi,
        bbox_inches="tight",
        pad_inches=0,
    )
    # plt.show()
    plt.close("all")


def plot_peak_regions(
    names_restricted,
    surname,
    imgs,
    zmaps,
    masker,
    threshold=90,
    saving_folder="./",
    ref_img=None,
    vmax=1,
):
    """Extract the peak regions of syntax and semantics and plot their superimposition."""
    categorical_values = True
    inflated = False
    hemispheres = ["left", "right"]
    views = ["lateral", "medial"]

    new_cmap = concat_colormaps(*["Reds", "Greens", yellow])
    plot_colorbar(new_cmap, vmax=0.1, vmin=0)

    titles = [f"{surname}-Syntax", f"{surname}-Semantics"]

    # Computing masks
    significance_masks = []
    fdr_ths = []
    for zmap in zmaps:
        mask, fdr_th = mask_from_zmap(zmap, ref_img)
        significance_masks.append(mask)
        fdr_ths.append(fdr_th)
    overlap_masks = [
        create_mask_from_threshold(
            img, masker, threshold=threshold, above=True, percentile=True
        )
        for img in imgs
    ]
    overlap_masks[0] = math_img(
        "img1*img2", img1=overlap_masks[0], img2=significance_masks[0]
    )
    overlap_masks[1] = math_img(
        "img1*img2", img1=overlap_masks[1], img2=significance_masks[1]
    )

    for i, img in enumerate(imgs):
        plot_name = (
            f"syntax_R_scores_distribution_{surname}_compared_to_BF"
            if i % 2 == 0
            else f"semantic_R_scores_distribution_{surname}_compared_to_BF"
        )
        # plot_distrib(img, syntax=(i%2==0), plot_name=plot_name, saving_folder=saving_folder, fdr_th=fdr_ths[i])
        plot_distrib(
            img,
            masker,
            syntax=(i % 2 == 0),
            plot_name=plot_name,
            saving_folder=saving_folder,
            fdr_th=None,
        )

    for index, mask in enumerate(overlap_masks):
        print(f"Using {np.sum(mask.get_fdata())} voxels in the mask of ", titles[index])

    # Computing overlap percentage
    pairs_overlap_syn_sem_core_indexes = itertools.combinations(
        np.arange(len(overlap_masks)), 2
    )
    for pair in pairs_overlap_syn_sem_core_indexes:
        print(
            f"Overlap between {titles[pair[0]]} and {titles[pair[1]]}: {masks_overlap(overlap_masks[pair[0]], overlap_masks[pair[1]])}"
        )

    # Computing Global Overlap
    overlap_syn_sem_raw = overlap_masks[0]
    for mask in overlap_masks[1:]:
        overlap_syn_sem_raw = intersect_binary(overlap_syn_sem_raw, mask)

    # Computing Global overlap percentage
    for index, mask in enumerate(overlap_masks):
        print(
            f"Using {np.sum(overlap_syn_sem_raw.get_fdata())} voxels in mask intersection."
        )
        print(
            f"Which means a global intersection of {100*np.sum(overlap_syn_sem_raw.get_fdata())/np.sum(mask.get_fdata())}"
        )

    surf_projs_syn_sem_core_superimposed = compute_surf_proj(
        overlap_masks,
        zmaps=None,
        masks=None,
        ref_img=ref_img,
        names=names_restricted,
        categorical_values=None,
        inflated=False,
        hemispheres=["left", "right"],
        views=["lateral", "medial"],
        kind="line",
        template=None,
    )

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
                cmap=new_cmap,
                inflated=inflated,
                threshold=1e-15,
                colorbar=False,
                symmetric_cbar=False,
                template=None,
                figure=figure,
                ax=ax,
                vmax=vmax,
            )

            surf_imgs = [
                surf_projs_syn_sem_core_superimposed[name][f"{hemi}-{view}"]
                for name in names_restricted
            ]
            d = [
                np.hstack(
                    [
                        surf_projs_syn_sem_core_superimposed[name][f"{hemi}-lateral"],
                        surf_projs_syn_sem_core_superimposed[name][f"{hemi}-medial"],
                    ]
                )
                for name in names_restricted
            ]
            print(names_restricted, len(d), [m.shape for m in d])
            for k, s in enumerate(d):
                d[k][np.isnan(s)] = 0
                d[k][s > 0] = 1
            print(
                f"hemi: {hemi} --- Jaccard score: {jaccard_score(d[0]>0, d[1]>0, average='binary')}"
            )

            for k, surf_img in enumerate(surf_imgs):
                surf_imgs[k][np.isnan(surf_img)] = 0
            surf_img = surf_imgs[0] / 3 + 2 * surf_imgs[1] / 3 - 0.1
            surf_img[surf_img == -0.1] = 0
            print(np.unique(surf_img))
            for i in np.unique(surf_img):
                print(
                    "-->",
                    hemi,
                    "-",
                    view,
                    "-",
                    i,
                    "-",
                    100 * np.sum(surf_img == i) / 10242,
                )
            if categorical_values:
                plotting.plot_surf_roi(roi_map=surf_img, **kwargs)
            else:
                plotting.plot_surf_stat_map(stat_map=surf_img, **kwargs)
            plot_name = f"Neurips_{hemi}_{view}_{surname}_10_percent_highest"
            format_figure = "pdf"
            dpi = 300
            plt.savefig(
                os.path.join(saving_folder, f"{plot_name}.{format_figure}"),
                format=format_figure,
                dpi=dpi,
                bbox_inches="tight",
                pad_inches=0,
            )
            plt.show()


def subj_peak_regions(
    names_restricted,
    surname,
    imgs,
    masker,
    subj,
    threshold=90,
    saving_folder="./",
    ref_img=None,
    vmax=1,
):
    """Idem but at subject level."""
    categorical_values = True
    inflated = False
    hemispheres = ["left", "right"]
    views = ["lateral", "medial"]

    new_cmap = concat_colormaps(*["Reds", "Greens", yellow])
    # plot_colorbar(new_cmap, vmax=0.1, vmin=0)

    titles = [f"{surname}-Syntax", f"{surname}-Semantics"]

    # Computing masks
    overlap_masks = [
        create_mask_from_threshold(
            img, masker, threshold=threshold, above=True, percentile=True
        )
        for img in imgs
    ]
    for i, img in enumerate(imgs):
        plot_name = (
            f"{subj}_syntax_R_scores_distribution_{surname}_compared_to_BF"
            if i % 2 == 0
            else f"{subj}_semantic_R_scores_distribution_{surname}_compared_to_BF"
        )
        # plot_distrib(img, syntax=(i%2==0), plot_name=plot_name, saving_folder=saving_folder, fdr_th=fdr_ths[i])
        plot_distrib(
            img,
            masker,
            syntax=(i % 2 == 0),
            plot_name=plot_name,
            saving_folder=saving_folder,
            fdr_th=None,
        )

    # for index, mask in enumerate(overlap_masks):
    #    print(f"Using {np.sum(mask.get_fdata())} voxels in the mask of ", titles[index])

    # Computing overlap percentage
    pairs_overlap_syn_sem_core_indexes = itertools.combinations(
        np.arange(len(overlap_masks)), 2
    )
    # for pair in pairs_overlap_syn_sem_core_indexes:
    #    print(
    #        f"Overlap between {titles[pair[0]]} and {titles[pair[1]]}: {masks_overlap(overlap_masks[pair[0]], overlap_masks[pair[1]])}"
    #    )

    # Computing Global Overlap
    overlap_syn_sem_raw = overlap_masks[0]
    for mask in overlap_masks[1:]:
        overlap_syn_sem_raw = intersect_binary(overlap_syn_sem_raw, mask)

    # Computing Global overlap percentage
    # for index, mask in enumerate(overlap_masks):
    #    print(
    #        f"Using {np.sum(overlap_syn_sem_raw.get_fdata())} voxels in mask intersection."
    #    )
    #    print(
    #        f"Which means a global intersection of {100*np.sum(overlap_syn_sem_raw.get_fdata())/np.sum(mask.get_fdata())}"
    #    )

    surf_projs_syn_sem_core_superimposed = compute_surf_proj(
        overlap_masks,
        zmaps=None,
        masks=None,
        ref_img=ref_img,
        names=names_restricted,
        categorical_values=None,
        inflated=False,
        hemispheres=["left", "right"],
        views=["lateral", "medial"],
        kind="line",
        template=None,
    )

    values = {"left": None, "right": None}
    distributions = {
        "left": {"lateral": None, "medial": None},
        "right": {"lateral": None, "medial": None},
    }
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
                cmap=new_cmap,
                inflated=inflated,
                threshold=1e-15,
                colorbar=False,
                symmetric_cbar=False,
                template=None,
                figure=figure,
                ax=ax,
                vmax=vmax,
            )

            surf_imgs = [
                surf_projs_syn_sem_core_superimposed[name][f"{hemi}-{view}"]
                for name in names_restricted
            ]
            d = [
                np.hstack(
                    [
                        surf_projs_syn_sem_core_superimposed[name][f"{hemi}-lateral"],
                        surf_projs_syn_sem_core_superimposed[name][f"{hemi}-medial"],
                    ]
                )
                for name in names_restricted
            ]
            for k, s in enumerate(d):
                d[k][np.isnan(s)] = 0
                d[k][s > 0] = 1
            values[hemi] = jaccard_score(d[0] > 0, d[1] > 0, average="binary")

            for k, surf_img in enumerate(surf_imgs):
                surf_imgs[k][np.isnan(surf_img)] = 0
            surf_img = surf_imgs[0] / 3 + 2 * surf_imgs[1] / 3 - 0.1
            surf_img[surf_img == -0.1] = 0
            distributions[hemi][view] = [
                100 * np.sum(surf_img == i) / 10242 for i in np.unique(surf_img)
            ]
            if categorical_values:
                plotting.plot_surf_roi(roi_map=surf_img, **kwargs)
            else:
                plotting.plot_surf_stat_map(stat_map=surf_img, **kwargs)
            plot_name = f"{subj}_Neurips_{hemi}_{view}_{surname}_10_percent_highest"
            format_figure = "pdf"
            dpi = 300
            plt.savefig(
                os.path.join(saving_folder, f"{plot_name}.{format_figure}"),
                format=format_figure,
                dpi=dpi,
                bbox_inches="tight",
                pad_inches=0,
            )
            # plt.show()
            plt.close("all")

    print(subj, "-", values)
    return values, distributions
