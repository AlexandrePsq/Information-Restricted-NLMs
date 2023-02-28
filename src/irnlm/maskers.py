import numpy as np
from nilearn import image

def masks_overlap(mask1, mask2):
    """Return the percentage of overlap between two mask.
    Args:
        - mask1: Nifti-Image
        - maks2: Nifti-Image
    Returns:
        - overlap: float
    """
    overlap = np.sum(mask1.get_fdata()*mask2.get_fdata())/np.sum(mask1.get_fdata())
    return overlap

def intersect_binary(img1, img2):
    """ Compute the intersection of two binary nifti images.
    Arguments:
        - img1: NifitImage
        - img2: NifitImage
    Returns:
        - intersection: NifitImage
    """
    intersection = image.math_img('img==2', img=image.math_img('img1+img2', img1=img1, img2=img2))
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


