import os
import numpy as np
import matplotlib.pyplot as plt

from irnlm.utils import check_folder
from fmri_encoder.data import fetch_masker
from fmri_encoder.plotting import compute_surf_proj
from irnlm.contrasts import plot_brain_fit


if __name__=='__main__':

    ###
    # Replicating Fig7: Context
    ###
    saving_folder = 'derivatives/Fig7'
    check_folder(saving_folder)
    masker = fetch_masker('derivatives/masker', None, **{'detrend': False, 'standardize': False})
    imgs = {}

    # Short
    name0 = "GPT-2_L-4_H-768_context-5_default_tokenizer_epoch-4_split-1_norm-None_temporal-shifting-0_{}_hidden-layer-3cv-alpha_-3_to_4_no-centering_wordrate-rms_chris-log_frequency"
    name1 = "GloVecv-alpha_-3_to_4wordrate-rms_chris-log_frequency_{}"
    effmaps_dict, zmaps_dict = plot_brain_fit(
        name0, name1, vmax=0.03, cmap='cold_hot', surname='GPT-2_context-5-vs-GloVe', 
        masker=masker, height_control='fdr', alpha=0.005, saving_folder=saving_folder)
    imgs['GPT-2_context-5-vs-GloVe'] = [effmaps_dict, zmaps_dict]
    effmaps_dict, zmaps_dict = plot_brain_fit(
        name0, name1, vmax=0.03, cmap='cold_hot', surname='GPT-2_context-5-vs-GloVe_unthresholded', 
        masker=masker, height_control='fdr', alpha=0.005, saving_folder=saving_folder,
        show_significant=False)
    
    # Medium
    name0 = "GPT-2_L-4_H-768_context-15_default_tokenizer_epoch-4_split-1_norm-None_temporal-shifting-0_{}_hidden-layer-3cv-alpha_-3_to_4_no-centering_wordrate-rms_chris-log_frequency"
    name1 = "GPT-2_L-4_H-768_context-5_default_tokenizer_epoch-4_split-1_norm-None_temporal-shifting-0_{}_hidden-layer-3cv-alpha_-3_to_4_no-centering_wordrate-rms_chris-log_frequency"
    effmaps_dict, zmaps_dict = plot_brain_fit(
        name0, name1, vmax=0.013, cmap='cold_hot', surname='GPT-2_context-15-vs-GPT-2_context-5', 
        masker=masker, height_control='fdr', alpha=0.005, saving_folder=saving_folder)
    imgs['GPT-2_context-5-vs-GloVe'] = [effmaps_dict, zmaps_dict]
    effmaps_dict, zmaps_dict = plot_brain_fit(
        name0, name1, vmax=0.013, cmap='cold_hot', surname='GPT-2_context-15-vs-GPT-2_context-5_unthresholded', 
        masker=masker, height_control='fdr', alpha=0.005, saving_folder=saving_folder,
        show_significant=False)
    
    # Long
    name0 = "GPT-2_L-4_H-768_context-45_default_tokenizer_epoch-4_split-1_norm-None_temporal-shifting-0_{}_hidden-layer-3cv-alpha_-3_to_4_no-centering_wordrate-rms_chris-log_frequency"
    name1 = "GPT-2_L-4_H-768_context-15_default_tokenizer_epoch-4_split-1_norm-None_temporal-shifting-0_{}_hidden-layer-3cv-alpha_-3_to_4_no-centering_wordrate-rms_chris-log_frequency"
    effmaps_dict, zmaps_dict = plot_brain_fit(
        name0, name1, vmax=0.013, cmap='cold_hot', surname='GPT-2_context-45-vs-GPT-2_context-15', 
        masker=masker, height_control='fdr', alpha=0.005, saving_folder=saving_folder)
    imgs['GPT-2_context-45-vs-GPT-2_context-15'] = [effmaps_dict, zmaps_dict]
    effmaps_dict, zmaps_dict = plot_brain_fit(
        name0, name1, vmax=0.013, cmap='cold_hot', surname='GPT-2_context-45-vs-GPT-2_context-15_unthresholded', 
        masker=masker, height_control='fdr', alpha=0.005, saving_folder=saving_folder,
        show_significant=False)


    titles = [
        'GPT-2_context-5_vs_GloVe',
        'GPT-2_context-15_vs_GPT-2_context-5',
        'GPT-2_context-45_vs_GPT-2_context-15',   
    ]

    for i, title in enumerate(titles):
        img = imgs[title][0]
        zmap = imgs[title][1]
        key = title + '_thresholded_fdr_0005'
        vmax = np.max(img.get_fdata())
        dict_ = {key: {'cmap': 'cold_hot', 'vmax': vmax}}
        print(key, vmax)
        surnames = list(dict_.keys())
        format_figure = 'pdf'
        dpi = 300

        surf_projs_raw_superimposed = compute_surf_proj(
            [img], 
            zmaps=[zmap], 
            masks=None, 
            ref_img=masker.mask_img_, 
            names=surnames, 
            categorical_values=None, 
            inflated=False, 
            hemispheres=['left', 'right'], 
            views=['lateral', 'medial'], 
            kind='line', 
            template=None,
            height_control='fdr',
            alpha=0.005, 
        )
        ll = surf_projs_raw_superimposed[key]['left-lateral']
        lm = surf_projs_raw_superimposed[key]['left-medial']
        rl = surf_projs_raw_superimposed[key]['right-lateral']
        rm = surf_projs_raw_superimposed[key]['right-medial']
        pre = np.hstack([ll, lm])
        post = np.hstack([rl, rm])
        pre = pre[~np.isnan(pre)]
        post = post[~np.isnan(post)]
        print(len(pre), len(post))

        plt.hist(pre, color='blue', alpha=0.5, label=f'left: {len(pre)} voxels', bins=np.linspace(-0.008, 0.03))
        plt.hist(post, color='red', alpha=0.5, label=f'right: {len(post)} voxels', bins=np.linspace(-0.008, 0.03))
        plt.xlabel('R score', fontsize=25)
        plt.ylabel('Number of voxels', fontsize=25)
        plt.xticks([0, 0.01, 0.02, 0.03], [0, 0.01, 0.02, 0.03], fontsize=20)
        plt.yticks([0, 500, 1000, 1500], [0, 500, 1000, 1500], fontsize=20)
        plt.legend(fontsize=15)
        plt.savefig(os.path.join(saving_folder, f'{key}_distrib.pdf'), format='pdf', dpi='100', bbox_inches = 'tight', pad_inches = 0, )
        plt.show()
        