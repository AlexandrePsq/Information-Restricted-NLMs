



### UNIQUE SYNTAX GPT-2
name0 = "GPT-2_L-4_H-768_Syntax_context-full_end-epoch-4_split-3-GPT-2_L-4_H-768_semantic_context-full_end-epoch-4_split-3_norm-None_temporal-shifting-0_{}_hidden-layer-3cv-alpha_-3_to_4_no-centering_wordrate-rms_chris-log_frequency"
name1 = "GPT-2_L-4_H-768_semantic_context-full_end-epoch-4_split-3_norm-None_temporal-shifting-0_{}_hidden-layer-3cv-alpha_-3_to_4_no-centering_wordrate-rms_chris-log_frequency"
vmax = 0.06
zmaps_dict, effmaps_dict, masks_dict, vmax_list, fdr_th = compute_diff_group_level_maps(name0, name1)
plt.hist(masker.transform(list(effmaps_dict.values())[0]).reshape(-1), bins=200)
plt.show()
print(fdr_th)
zmaps_dict = {'GPT-2_Syntax+Semantic-vs-GPT-2_Semantic': list(zmaps_dict.values())[0]}
d = list(effmaps_dict.values())[0]
mask = math_img('img>0', img=d)
d = math_img('img1*img2', img1=d, img2=mask)
effmaps_dict = {'GPT-2_Syntax+Semantic-vs-GPT-2_Semantic': d}
dict_ = {'GPT-2_Syntax+Semantic-vs-GPT-2_Semantic': {'cmap': red, 'vmax': vmax, }}
plot_group_level_surface_maps(effmaps_dict, zmaps_dict, dict_, threshold=1e-15, height_control='fdr', alpha=0.005)

### UNIQUE SYNTAX  GLOVE
name0 = "GloVe_Syntax_full_GloVe_Semantic_cv-alpha_-3_to_4wordrate-rms_chris-log_frequency_{}"
name1 = "GloVe_Semanticcv-alpha_-3_to_4wordrate-rms_chris-log_frequency_{}"
vmax = 0.06
zmaps_dict, effmaps_dict, masks_dict, vmax_list, fdr_th = compute_diff_group_level_maps(name0, name1 )
plt.hist(masker.transform(list(effmaps_dict.values())[0]).reshape(-1), bins=200)
plt.show()
print(fdr_th)
zmaps_dict = {'Glove_Syntax+Semantic-vs-Glove_Semantic': list(zmaps_dict.values())[0]}
effmaps_dict = {'Glove_Syntax+Semantic-vs-Glove_Semantic': list(effmaps_dict.values())[0]}
dict_ = {'Glove_Syntax+Semantic-vs-Glove_Semantic': {'cmap': red, 'vmax': vmax, } }
plot_group_level_surface_maps(effmaps_dict, zmaps_dict, dict_, threshold=1e-15, height_control='fdr', alpha=0.005)


### UNIQUE SEMANTIC  GPT-2
name0 = "GPT-2_L-4_H-768_Syntax_context-full_end-epoch-4_split-3-GPT-2_L-4_H-768_semantic_context-full_end-epoch-4_split-3_norm-None_temporal-shifting-0_{}_hidden-layer-3cv-alpha_-3_to_4_no-centering_wordrate-rms_chris-log_frequency"
name1 = "GPT-2_L-4_H-768_Syntax_context-full_end-epoch-4_split-3_norm-None_temporal-shifting-0_{}_hidden-layer-3cv-alpha_-3_to_4_no-centering_wordrate-rms_chris-log_frequency"
vmax = 0.06
zmaps_dict, effmaps_dict, masks_dict, vmax_list, fdr_th = compute_diff_group_level_maps(name0, name1)
plt.hist(masker.transform(list(effmaps_dict.values())[0]).reshape(-1), bins=200)
plt.show()
print(fdr_th)
zmaps_dict = {'GPT-2_Syntax+Semantic-vs-GPT-2_Syntax': list(zmaps_dict.values())[0]}
effmaps_dict = {'GPT-2_Syntax+Semantic-vs-GPT-2_Syntax': list(effmaps_dict.values())[0]}
dict_ = {'GPT-2_Syntax+Semantic-vs-GPT-2_Syntax': {'cmap': green, 'vmax': vmax, } }
plot_group_level_surface_maps(effmaps_dict, zmaps_dict, dict_, threshold=1e-15, height_control='fdr', alpha=0.005)

### UNIQUE SEMANTIC  GLOVE
name0 = "GloVe_Syntax_full_GloVe_Semantic_cv-alpha_-3_to_4wordrate-rms_chris-log_frequency_{}"
name1 = "GloVe_Syntax_fullcv-alpha_-3_to_4wordrate-rms_chris-log_frequency_{}"
vmax = 0.06
zmaps_dict, effmaps_dict, masks_dict, vmax_list, fdr_th = compute_diff_group_level_maps(name0, name1)
plt.hist(masker.transform(list(effmaps_dict.values())[0]).reshape(-1), bins=200)
plt.show()
print(fdr_th)
zmaps_dict = {'Glove_Syntax+Semantic-vs-Glove_Syntax': list(zmaps_dict.values())[0]}
effmaps_dict = {'Glove_Syntax+Semantic-vs-Glove_Syntax': list(effmaps_dict.values())[0]}
dict_ = {'Glove_Syntax+Semantic-vs-Glove_Syntax': {'cmap': green, 'vmax': vmax, } }
plot_group_level_surface_maps(effmaps_dict, zmaps_dict, dict_, threshold=1e-15, height_control='fdr', alpha=0.005)


## SYNERGY GPT-2
name0 = "GPT-2_L-4_H-768_default_tokenizer_full_end-epoch-4_split-3_norm-None_temporal-shifting-0_{}_hidden-layer-3cv-alpha_-3_to_4_no-centering_wordrate-rms_chris-log_frequency"
name1 = "GPT-2_L-4_H-768_Syntax_context-full_end-epoch-4_split-3-GPT-2_L-4_H-768_semantic_context-full_end-epoch-4_split-3_norm-None_temporal-shifting-0_{}_hidden-layer-3cv-alpha_-3_to_4_no-centering_wordrate-rms_chris-log_frequency"
vmax = 0.1
zmaps_dict, effmaps_dict, masks_dict, vmax_list, fdr_th = compute_diff_group_level_maps(name0, name1)
plt.hist(masker.transform(list(effmaps_dict.values())[0]).reshape(-1), bins=200)
plt.show()
print(fdr_th)
zmaps_dict = {'GPT-2_full-vs-GPT-2_Sem+Syn_intertwined': list(zmaps_dict.values())[0]}
effmaps_dict = {'GPT-2_full-vs-GPT-2_Sem+Syn_intertwined': list(effmaps_dict.values())[0]}
dict_ = {'GPT-2_full-vs-GPT-2_Sem+Syn_intertwined': {'cmap': blue, 'vmax': vmax, } }
plot_group_level_surface_maps(effmaps_dict, zmaps_dict, dict_, threshold=1e-15, height_control='fdr', alpha=0.005)
## SYNERGY GLOVE
name0 = "GloVecv-alpha_-3_to_4wordrate-rms_chris-log_frequency_{}"
name1 = "GloVe_Syntax_full_GloVe_Semantic_cv-alpha_-3_to_4wordrate-rms_chris-log_frequency_{}"
vmax = 0.1
zmaps_dict, effmaps_dict, masks_dict, vmax_list, fdr_th = compute_diff_group_level_maps(name0, name1)
plt.hist(masker.transform(list(effmaps_dict.values())[0]).reshape(-1), bins=200)
plt.show()
print(fdr_th)
zmaps_dict = {'GloVe_full-vs-GloVe_Sem+Syn_intertwined': list(zmaps_dict.values())[0]}
effmaps_dict = {'GloVe_full-vs-GloVe_Sem+Syn_intertwined': list(effmaps_dict.values())[0]}
dict_ = {'GloVe_full-vs-GloVe_Sem+Syn_intertwined': {'cmap': blue, 'vmax': vmax, } }
plot_group_level_surface_maps(effmaps_dict, zmaps_dict, dict_, threshold=1e-15, height_control='fdr', alpha=0.005)
