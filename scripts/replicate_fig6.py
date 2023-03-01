


### Semantic GPT-2
name0 = "GPT-2_L-4_H-768_semantic_context-full_end-epoch-4_split-3_norm-None_temporal-shifting-0_{}_hidden-layer-3cv-alpha_-3_to_4_no-centering_wordrate-rms_chris-log_frequency"
name1 = "GloVe_Semanticcv-alpha_-3_to_4wordrate-rms_chris-log_frequency_{}"
vmax = 0.04
zmaps_dict, effmaps_dict, masks_dict, vmax_list, fdr_th = compute_diff_group_level_maps(name0, name1)
plt.hist(masker.transform(list(effmaps_dict.values())[0]).reshape(-1), bins=200)
plt.show()
print(fdr_th)
zmaps_dict = {'GPT-2_Semantic-vs-GloVe_Semantic': list(zmaps_dict.values())[0]}
effmaps_dict = {'GPT-2_Semantic-vs-GloVe_Semantic': list(effmaps_dict.values())[0]}
dict_ = {'GPT-2_Semantic-vs-GloVe_Semantic': {'cmap': 'cold_hot', 'vmax': vmax, } }
plot_group_level_surface_maps(effmaps_dict, zmaps_dict, dict_, threshold=1e-15, height_control='fdr', alpha=0.005)

### Syntax GPT-2
name0 = "GPT-2_L-4_H-768_Syntax_context-full_end-epoch-4_split-3_norm-None_temporal-shifting-0_{}_hidden-layer-3cv-alpha_-3_to_4_no-centering_wordrate-rms_chris-log_frequency"
name1 = "GloVe_Syntax_fullcv-alpha_-3_to_4wordrate-rms_chris-log_frequency_{}"
vmax = 0.04
dict_ = {key: {'cmap': 'cold_hot', 'vmax': vmax, } for key in effmaps_dict.keys()}
zmaps_dict, effmaps_dict, masks_dict, vmax_list, fdr_th = compute_diff_group_level_maps(name0, name1)
plt.hist(masker.transform(list(effmaps_dict.values())[0]).reshape(-1), bins=200)
plt.show()
print(fdr_th)
zmaps_dict = {'GPT-2_Syntax-vs-GloVe_Syntax': list(zmaps_dict.values())[0]}
effmaps_dict = {'GPT-2_Syntax-vs-GloVe_Syntax': list(effmaps_dict.values())[0]}
dict_ = {'GPT-2_Syntax-vs-GloVe_Syntax': {'cmap': 'cold_hot', 'vmax': vmax, } }
plot_group_level_surface_maps(effmaps_dict, zmaps_dict, dict_, threshold=1e-15, height_control='fdr', alpha=0.005)


### Full GPT-2
name0 = "GPT-2_L-4_H-768_default_tokenizer_full_end-epoch-4_split-3_norm-None_temporal-shifting-0_{}_hidden-layer-3cv-alpha_-3_to_4_no-centering_wordrate-rms_chris-log_frequency"
name1 = "GloVecv-alpha_-3_to_4wordrate-rms_chris-log_frequency_{}"
vmax = 0.04
dict_ = {key: {'cmap': 'cold_hot', 'vmax': vmax, } for key in effmaps_dict.keys()}
zmaps_dict, effmaps_dict, masks_dict, vmax_list, fdr_th = compute_diff_group_level_maps(name0, name1)
plt.hist(masker.transform(list(effmaps_dict.values())[0]).reshape(-1), bins=200)
plt.show()
print(fdr_th)
zmaps_dict = {'GPT-2_full-vs-GloVe_full': list(zmaps_dict.values())[0]}
effmaps_dict = {'GPT-2_full-vs-GloVe_full': list(effmaps_dict.values())[0]}
dict_ = {'GPT-2_full-vs-GloVe_full': {'cmap': 'cold_hot', 'vmax': vmax, } }
plot_group_level_surface_maps(effmaps_dict, zmaps_dict, dict_, threshold=1e-15, height_control='fdr', alpha=0.005)
