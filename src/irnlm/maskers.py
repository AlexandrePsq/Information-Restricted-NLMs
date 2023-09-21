import os
import numpy as np
import nibabel as nib
from nilearn import image, datasets, maskers
from nilearn.image import math_img, resample_to_img

from irnlm.utils import read_yaml, save_yaml, check_folder


def masks_overlap(mask1, mask2):
    """Return the percentage of overlap between two mask.
    Args:
        - mask1: Nifti-Image
        - maks2: Nifti-Image
    Returns:
        - overlap: float
    """
    overlap = np.sum(mask1.get_fdata() * mask2.get_fdata()) / np.sum(mask1.get_fdata())
    return overlap


def intersect_binary(img1, img2):
    """Compute the intersection of two binary nifti images.
    Arguments:
        - img1: NifitImage
        - img2: NifitImage
    Returns:
        - intersection: NifitImage
    """
    intersection = image.math_img(
        "img==2", img=image.math_img("img1+img2", img1=img1, img2=img2)
    )
    return intersection


def create_mask_from_threshold(img, masker, threshold=95, above=True, percentile=True):
    """Create a mask from voxels above (or below) a threshold.
    Args:
        - img: Nifti-image
        - threshold: float
        - masker: NifitMasker
        - above: bool
        - percentile: bool
    Returns:
        - img: Nifti-image
    """
    data = masker.transform(img)
    if percentile:
        threshold = np.percentile(data, threshold)
    if above:
        img = image.new_img_like(img, img.get_fdata() > threshold)
    else:
        img = image.new_img_like(img, img.get_fdata() < threshold)
    return img


def fetch_atlas_labels(
    atlas_type, n_rois=None, name="cort-maxprob-thr25-2mm", symmetric_split=True
):
    """For harvard-oxford atlas, we removed the ‘background‘."""
    if atlas_type == "harvard-oxford":
        atlas = datasets.fetch_atlas_harvard_oxford(
            name, symmetric_split=symmetric_split
        )
        labels = atlas["labels"]
        atlas_maps = image.load_img(atlas["maps"])
        labels = labels[1:]
    else:
        atlas_detailed = datasets.fetch_atlas_schaefer_2018(
            n_rois=n_rois,
            yeo_networks=17,
            resolution_mm=1,
            data_dir=None,
            base_url=None,
            resume=True,
            verbose=1,
        )

        labels = [roi.decode("utf-8") for roi in atlas_detailed["labels"]]
        atlas_maps = image.load_img(atlas_detailed["maps"])
    return atlas_maps, labels


def get_roi_mask(
    atlas_maps,
    index_mask,
    labels,
    path=None,
    global_mask=None,
    resample_to_img_=None,
    intersect_with_img=False,
    PROJECT_PATH="/Users/alexpsq/Code/Parietal/LePetitPrince/",
):
    """Return the Niftimasker object for a given ROI based on an atlas.
    Optionally resampled based on a resample_to_img_ image and another masker (global masker) parameters.
    """
    if path is None:
        check_folder(os.path.join(PROJECT_PATH, "derivatives/fMRI/ROI_masks"))
        path = os.path.join(
            PROJECT_PATH, "derivatives/fMRI/ROI_masks", labels[index_mask]
        )  # be careful to remove ‘background‘ from labels
    if os.path.exists(path + ".nii.gz") and os.path.exists(path + ".yml"):
        masker = load_masker(
            path,
            resample_to_img_=resample_to_img_,
            intersect_with_img=intersect_with_img,
        )
    else:
        mask = math_img("img=={}".format(index_mask + 1), img=atlas_maps)
        if resample_to_img_ is not None:
            mask = resample_to_img(mask, resample_to_img_, interpolation="nearest")
            if intersect_with_img:
                mask_img = math_img(
                    "img==2",
                    img=math_img("img1+img2", img1=mask_img, img2=resample_to_img_),
                )
        masker = maskers.NiftiMasker(mask)
        if global_mask:
            params = read_yaml(global_mask + ".yml")
            params["detrend"] = False
            params["standardize"] = False
            masker.set_params(**params)
        masker.fit()
        save_masker(masker, path)
    return masker


def load_masker(path, resample_to_img_=None, intersect_with_img=False, **kwargs):
    """Given a path without the extension, load the associated yaml anf Nifti files to compute
    the associated masker.
    Arguments:
        - path: str
        - resample_to_img_: Nifti image (optional)
        - intersect_with_img: bool (optional)
        - kwargs: dict
    """
    params = read_yaml(path + ".yml")
    mask_img = nib.load(path + ".nii.gz")
    if resample_to_img_ is not None:
        mask_img = image.resample_to_img(
            mask_img, resample_to_img_, interpolation="nearest"
        )
        if intersect_with_img:
            mask_img = intersect_binary(mask_img, resample_to_img_)
    masker = maskers.NiftiMasker(mask_img)
    masker.set_params(**params)
    if kwargs:
        masker.set_params(**kwargs)
    masker.fit()
    return masker


def save_masker(masker, path):
    """Save the yaml file and image associated with a masker"""
    params = masker.get_params()
    params = {
        key: params[key]
        for key in [
            "detrend",
            "dtype",
            "high_pass",
            "low_pass",
            "mask_strategy",
            "memory_level",
            "smoothing_fwhm",
            "standardize",
            "t_r",
            "verbose",
        ]
    }
    nib.save(masker.mask_img_, path + ".nii.gz")
    save_yaml(params, path + ".yml")
