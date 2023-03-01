

## Full GPT-2
name0 = "GPT-2_L-4_H-768_context-5_default_tokenizer_epoch-4_split-1_norm-None_temporal-shifting-0_{}_hidden-layer-3cv-alpha_-3_to_4_no-centering_wordrate-rms_chris-log_frequency"
name1 = "GloVecv-alpha_-3_to_4wordrate-rms_chris-log_frequency_{}"
zmaps_dict, effmaps_dict, masks_dict, vmax_list, fdr_th = compute_diff_group_level_maps(name0, name1)
plt.hist(masker.transform(list(effmaps_dict.values())[0]).reshape(-1), bins=200)
plt.xlim((-0.01, 0.03))
plt.show()
print(fdr_th)
zmaps_dict = {'GPT-2_context-5-vs-GloVe_unthresholded': list(zmaps_dict.values())[0]}
effmaps_dict = {'GPT-2_context-5-vs-GloVe_unthresholded': list(effmaps_dict.values())[0]}
vmax = np.max(list(effmaps_dict.values())[0].get_fdata())
short_img = list(effmaps_dict.values())[0]
short_mask = mask_from_zmap(list(zmaps_dict.values())[0], ref_img)[0]
print(vmax)
dict_ = {'GPT-2_context-5-vs-GloVe_unthresholded': {'cmap': 'cold_hot', 'vmax': vmax, } }
plot_group_level_surface_maps(effmaps_dict, zmaps_dict, dict_, threshold=1e-15, height_control='fdr', alpha=0.005)


## Full GPT-2
name0 = "GPT-2_L-4_H-768_context-15_default_tokenizer_epoch-4_split-1_norm-None_temporal-shifting-0_{}_hidden-layer-3cv-alpha_-3_to_4_no-centering_wordrate-rms_chris-log_frequency"
name1 = "GPT-2_L-4_H-768_context-5_default_tokenizer_epoch-4_split-1_norm-None_temporal-shifting-0_{}_hidden-layer-3cv-alpha_-3_to_4_no-centering_wordrate-rms_chris-log_frequency"
zmaps_dict, effmaps_dict, masks_dict, vmax_list, fdr_th = compute_diff_group_level_maps(name0, name1)
plt.hist(masker.transform(list(effmaps_dict.values())[0]).reshape(-1), bins=200)
plt.xlim((-0.01, 0.03))
plt.show()
print(fdr_th)
zmaps_dict = {'GPT-2_context-15-vs-GPT-2_context-5_bonf': list(zmaps_dict.values())[0]}
effmaps_dict = {'GPT-2_context-15-vs-GPT-2_context-5_bonf': list(effmaps_dict.values())[0]}
vmax = np.max(list(effmaps_dict.values())[0].get_fdata())
medium_img = list(effmaps_dict.values())[0]
medium_mask = mask_from_zmap(list(zmaps_dict.values())[0], ref_img)[0]
print(vmax)
dict_ = {'GPT-2_context-15-vs-GPT-2_context-5_bonf': {'cmap': 'cold_hot', 'vmax': vmax, } }
plot_group_level_surface_maps(effmaps_dict, zmaps_dict, dict_, threshold=1e-15, height_control='fdr', alpha=0.005)


## Full GPT-2
name0 = "GPT-2_L-4_H-768_context-45_default_tokenizer_epoch-4_split-1_norm-None_temporal-shifting-0_{}_hidden-layer-3cv-alpha_-3_to_4_no-centering_wordrate-rms_chris-log_frequency"
name1 = "GPT-2_L-4_H-768_context-15_default_tokenizer_epoch-4_split-1_norm-None_temporal-shifting-0_{}_hidden-layer-3cv-alpha_-3_to_4_no-centering_wordrate-rms_chris-log_frequency"
zmaps_dict, effmaps_dict, masks_dict, vmax_list, fdr_th = compute_diff_group_level_maps(name0, name1)
plt.hist(masker.transform(list(effmaps_dict.values())[0]).reshape(-1), bins=200)
plt.xlim((-0.01, 0.03))
plt.show()
print(fdr_th)
zmaps_dict = {'GPT-2_context-45-vs-GPT-2_context-15_bonf': list(zmaps_dict.values())[0]}
effmaps_dict = {'GPT-2_context-45-vs-GPT-2_context-15_bonf': list(effmaps_dict.values())[0]}
vmax = np.max(list(effmaps_dict.values())[0].get_fdata())
long_img = list(effmaps_dict.values())[0]
long_mask = mask_from_zmap(list(zmaps_dict.values())[0], ref_img)[0]
print(vmax)
dict_ = {'GPT-2_context-45-vs-GPT-2_context-15_bonf': {'cmap': 'cold_hot', 'vmax': vmax, } }
plot_group_level_surface_maps(effmaps_dict, zmaps_dict, dict_, threshold=1e-15, height_control='fdr', alpha=0.005)



names = [
    "GloVecv-alpha_-3_to_4wordrate-rms_chris-log_frequency_{}",
    "GPT-2_L-4_H-768_context-5_default_tokenizer_epoch-4_split-1_norm-None_temporal-shifting-0_{}_hidden-layer-3cv-alpha_-3_to_4_no-centering_wordrate-rms_chris-log_frequency",
    "GPT-2_L-4_H-768_context-15_default_tokenizer_epoch-4_split-1_norm-None_temporal-shifting-0_{}_hidden-layer-3cv-alpha_-3_to_4_no-centering_wordrate-rms_chris-log_frequency",
    "GPT-2_L-4_H-768_context-45_default_tokenizer_epoch-4_split-1_norm-None_temporal-shifting-0_{}_hidden-layer-3cv-alpha_-3_to_4_no-centering_wordrate-rms_chris-log_frequency",
]
zmaps_dict1, effmaps_dict1, masks_dict1, vmax_list1, fdr_th = compute_diff_group_level_maps(
    names[1],
    names[0]
)
print(fdr_th)
zmaps_dict2, effmaps_dict2, masks_dict2, vmax_list2, fdr_th = compute_diff_group_level_maps(
    names[2],
    names[1]
)
print(fdr_th)
zmaps_dict3, effmaps_dict3, masks_dict3, vmax_list3, fdr_th = compute_diff_group_level_maps(
    names[3],
    names[2]
)
print(fdr_th)

from scipy import stats

titles = [
    'GPT-2_context-5_vs_GloVe',
    'GPT-2_context-15_vs_GPT-2_context-5',
    'GPT-2_context-45_vs_GPT-2_context-15',   
]
dictionary_ = {}

for i, (img, zmap) in enumerate(zip([effmaps_dict1, effmaps_dict2, effmaps_dict3], [zmaps_dict1, zmaps_dict2, zmaps_dict3])):
    key = list(img.keys())[0]
    img = img[key]
    zmap = zmap[key]
    key = titles[i] + '_thresholded_fdr_0005'
    vmax = np.max(img.get_fdata())
    dict_ = {key: {'cmap': 'cold_hot', 'vmax': vmax}}
    print(key)
    print(vmax)
    surnames = list(dict_.keys())
    corr_part_imgs_ = [img]
    corr_part_zmaps_ = [zmap]
#
    format_figure = 'pdf'
    dpi = 300
#
    surf_projs_raw_superimposed = compute_surf_proj(
        corr_part_imgs_, 
        zmaps=corr_part_zmaps_, 
        masks=None, 
        ref_img=ref_img, 
        names=surnames, 
        categorical_values=None, 
        inflated=False, hemispheres=['left', 'right'], 
        views=['lateral', 'medial'], kind='line', template=None,
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
    print('TEST:', stats.ks_2samp(pre, post, alternative='two-sided'))
    print('TEST:', stats.ks_2samp(pre, post, alternative='greater'))
    print('TEST:', stats.ks_2samp(pre, post, alternative='less'), '\n')
    
    plt.hist(pre, color='blue', alpha=0.5, label=f'left: {len(pre)} voxels', bins=np.linspace(-0.008, 0.03))
    plt.hist(post, color='red', alpha=0.5, label=f'right: {len(post)} voxels', bins=np.linspace(-0.008, 0.03))
    plt.xlabel('R score', fontsize=25)
    plt.ylabel('Number of voxels', fontsize=25)
    plt.xticks([0, 0.01, 0.02, 0.03], [0, 0.01, 0.02, 0.03], fontsize=20)
    plt.yticks([0, 500, 1000, 1500], [0, 500, 1000, 1500], fontsize=20)
    plt.legend(fontsize=15)
    plt.savefig(os.path.join(saving_folder, f'{key}_distrib.pdf'), format='pdf', dpi='100', bbox_inches = 'tight', pad_inches = 0, )
    plt.show()
    
    if i==0:
        name='short'
    elif i==1:
        name='medium'
    else:
        name='long'
    dictionary_[name+'_left'] = pre
    dictionary_[name+'_right'] = post
    