import numpy as np
from nilearn.image import concat_imgs, math_img
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


from fmri_encoder.plotting import (
    plot_colorbar, 
    concat_colormaps
)
from irnlm.utils import check_folder
from fmri_encoder.data import fetch_masker
from irnlm.data.utils import fetch_model_maps
from irnlm.specificity_index import specificity_index
from irnlm.contrasts import (
    compute_diff_imgs, 
    group_analysis, 
    plot_group_level_surface_maps
)

# Defniinf Colormaps
cmap = plt.cm.get_cmap('black_red')
cmap = cmap(np.linspace(0, 1, 350))
cmap = ListedColormap(cmap[70:280])
cmap2 = plt.cm.get_cmap('black_green_r')
cmap2 = cmap2(np.linspace(0, 1, 350))
cmap2 = ListedColormap(cmap2[70:280])
cmap4 = concat_colormaps(*[cmap, cmap2], cutting_threshold=0)
plot_colorbar(cmap4)


# Definin paths and masker
saving_folder = 'derivatives/Fig4/'
check_folder(saving_folder)
masker = fetch_masker('derivatives/masker', None, **{'detrend': False, 'standardize': False})
ref_img = masker.mask_img_

names = [
    "BF_wordrate-rms_chris-log_frequency_cv-alpha_-3_to_4__{}",
    "GloVe_Semanticcv-alpha_-3_to_4wordrate-rms_chris-log_frequency_{}",
    "GloVe_Syntax_fullcv-alpha_-3_to_4wordrate-rms_chris-log_frequency_{}",
    "GPT-2_L-4_H-768_Syntax_context-full_end-epoch-4_split-3_norm-None_temporal-shifting-0_{}_hidden-layer-3cv-alpha_-3_to_4_no-centering_wordrate-rms_chris-log_frequency",
    "GPT-2_L-4_H-768_semantic_context-full_end-epoch-4_split-3_norm-None_temporal-shifting-0_{}_hidden-layer-3cv-alpha_-3_to_4_no-centering_wordrate-rms_chris-log_frequency",
]


# Fetching data
data = fetch_model_maps(
        names, 
        language='english',
        verbose=0,
        FMRIDATA_PATH="derivatives/fMRI/maps/english"
    )
Baseline = data["BF_wordrate-rms_chris-log_frequency_cv-alpha_-3_to_4_"]['Pearson_coeff']
Glove_Syntax_maps = data["GloVe_Syntax_fullcv-alpha_-3_to_4wordrate-rms_chris-log_frequency"]['Pearson_coeff']
Glove_Semantics_maps = data["GloVe_Semanticcv-alpha_-3_to_4wordrate-rms_chris-log_frequency"]['Pearson_coeff']
GPT2_Syntax_maps = data["GPT-2_L-4_H-768_Syntax_context-full_end-epoch-4_split-3_norm-None_temporal-shifting-0_hidden-layer-3cv-alpha_-3_to_4_no-centering_wordrate-rms_chris-log_frequency"]['Pearson_coeff']
GPT2_Semantics_maps = data["GPT-2_L-4_H-768_semantic_context-full_end-epoch-4_split-3_norm-None_temporal-shifting-0_hidden-layer-3cv-alpha_-3_to_4_no-centering_wordrate-rms_chris-log_frequency"]['Pearson_coeff']


# Computing differences between images
diff_imgs_glove_syntax = compute_diff_imgs(Glove_Syntax_maps, Baseline)
diff_imgs_glove_semantic = compute_diff_imgs(Glove_Semantics_maps, Baseline)
diff_imgs_gpt2_syntax = compute_diff_imgs(GPT2_Syntax_maps, Baseline)
diff_imgs_gpt2_semantic = compute_diff_imgs(GPT2_Semantics_maps, Baseline)


# Computing specificity indexes
imgs_glove = [
    specificity_index(
        img_sem, 
        img_syn, 
        masker, 
        threshold=3
        ) for (img_sem, img_syn) in zip(diff_imgs_glove_semantic, diff_imgs_glove_syntax)]
imgs_gpt2 = [
    specificity_index(
        img_sem, 
        img_syn, 
        masker, 
        threshold=3
        ) for (img_sem, img_syn) in zip(diff_imgs_gpt2_semantic, diff_imgs_gpt2_syntax)]


# Computing group-level analysis
# The significance masks assess in which brain regions did semantic or syntactic
# features significantly fitted brain data. (Top row of figure 4.)
_, _, mask_glove_syntax , fdr_th = group_analysis(
    diff_imgs_glove_syntax, 
    smoothing_fwhm=6, 
    vmax=None, 
    design_matrix=None, 
    p_val=None, 
    fdr=0.005, 
    mask_img=masker.mask_img_, 
    cluster_threshold=30, 
    height_control='fdr'
)
_, _, mask_glove_semantics , fdr_th = group_analysis(
    diff_imgs_glove_semantic, 
    smoothing_fwhm=6, 
    vmax=None, 
    design_matrix=None, 
    p_val=None, 
    fdr=0.005, 
    mask_img=masker.mask_img_, 
    cluster_threshold=30, 
    height_control='fdr'
)
_, _, mask_gpt2_syntax , fdr_th = group_analysis(
    diff_imgs_gpt2_syntax, 
    smoothing_fwhm=6, 
    vmax=None, 
    design_matrix=None, 
    p_val=None, 
    fdr=0.005, 
    mask_img=masker.mask_img_, 
    cluster_threshold=30, 
    height_control='fdr'
)
_, _, mask_gpt2_semantics , fdr_th = group_analysis(
    diff_imgs_gpt2_semantic, 
    smoothing_fwhm=6, 
    vmax=None, 
    design_matrix=None, 
    p_val=None, 
    fdr=0.005, 
    mask_img=masker.mask_img_, 
    cluster_threshold=30,
    height_control='fdr'
)

mask_glove = math_img('img1+img2 > 0', img1=mask_glove_syntax, img2=mask_glove_semantics)
mask_gpt2 = math_img('img1+img2 > 0', img1=mask_gpt2_syntax, img2=mask_gpt2_semantics)


# We stack specificity index maps
img_glove = concat_imgs(imgs_glove)
img_gpt2 = concat_imgs(imgs_gpt2)


# Computign significativity on specificity index maps
zmap_glove, effmap_glove, mask_glove_ttest, fdr_th = group_analysis(
            imgs_glove, 
            smoothing_fwhm=6, 
            design_matrix=None, 
            p_val=None, 
            fdr=0.005, 
            mask_img=math_img('img1*img2', img1=masker.mask_img_, img2=mask_glove),
            cluster_threshold=30, 
            height_control='fdr')
print(fdr_th)

zmap_gpt2, effmap_gpt2, mask_gpt2_ttest, fdr_th = group_analysis(
            imgs_gpt2, 
            smoothing_fwhm=6, 
            design_matrix=None, 
            p_val=None, 
            fdr=0.005, 
            mask_img=math_img('img1*img2', img1=masker.mask_img_, img2=mask_gpt2),
            cluster_threshold=30, 
            height_control='fdr')
print(fdr_th)


# Plotting
effmaps_dict = {
    'glove_ratio_ttest_fdr_005': effmap_glove, 
    'gpt2_ratio_ttest_fdr_005': effmap_gpt2,
}
zmaps_dict = {
    'glove_ratio_ttest_fdr_005': mask_glove_ttest,
    'gpt2_ratio_ttest_fdr_005': mask_gpt2_ttest,
}
dict_ = {
    'glove_ratio_ttest_fdr_005':  {'cmap': cmap4, 'vmax': 1}, 
    'gpt2_ratio_ttest_fdr_005': {'cmap': cmap4, 'vmax': 1}
}
plot_group_level_surface_maps(
        effmaps_dict, 
        zmaps_dict, 
        dict_, 
        ref_img=masker.mask_img_,
        threshold=1e-15, 
        height_control='fdr', 
        alpha=0.005, 
        saving_folder=saving_folder
    )


effmaps_dict = {
    'glove_ratio': effmap_glove, 
    'gpt2_ratio': effmap_gpt2,
}
zmaps_dict = {
    'glove_ratio': mask_glove,
    'gpt2_ratio': mask_gpt2,
}
dict_ = {
    'glove_ratio':  {'cmap': cmap4, 'vmax': 1}, 
    'gpt2_ratio': {'cmap': cmap4, 'vmax': 1}
}
plot_group_level_surface_maps(
        effmaps_dict, 
        zmaps_dict, 
        dict_, 
        ref_img=masker.mask_img_,
        threshold=1e-15, 
        height_control='fdr', 
        alpha=0.005, 
        saving_folder=saving_folder
    )
