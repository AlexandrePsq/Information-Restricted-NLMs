


from nilearn.image import concat_imgs
names = [
    "BF_wordrate-rms_chris-log_frequency_cv-alpha_-3_to_4__{}",
    "GloVe_Semanticcv-alpha_-3_to_4wordrate-rms_chris-log_frequency_{}",
    "GloVe_Syntax_fullcv-alpha_-3_to_4wordrate-rms_chris-log_frequency_{}",
    "GPT-2_L-4_H-768_Syntax_context-full_end-epoch-4_split-3_norm-None_temporal-shifting-0_{}_hidden-layer-3cv-alpha_-3_to_4_no-centering_wordrate-rms_chris-log_frequency",
    "GPT-2_L-4_H-768_semantic_context-full_end-epoch-4_split-3_norm-None_temporal-shifting-0_{}_hidden-layer-3cv-alpha_-3_to_4_no-centering_wordrate-rms_chris-log_frequency",
]
data = dataset_utils.fetch_model_maps(
        names, 
        language='english',
        verbose=0,
        FMRIDATA_PATH="/neurospin/unicog/protocols/IRMf/LePetitPrince_Pallier_2018/LePetitPrince/derivatives/fMRI/maps/english"
    )


Baseline = data["BF_wordrate-rms_chris-log_frequency_cv-alpha_-3_to_4_"]['Pearson_coeff']
Glove_Syntax_maps = data["GloVe_Syntax_fullcv-alpha_-3_to_4wordrate-rms_chris-log_frequency"]['Pearson_coeff']
Glove_Semantics_maps = data["GloVe_Semanticcv-alpha_-3_to_4wordrate-rms_chris-log_frequency"]['Pearson_coeff']
GPT2_Syntax_maps = data["GPT-2_L-4_H-768_Syntax_context-full_end-epoch-4_split-3_norm-None_temporal-shifting-0_hidden-layer-3cv-alpha_-3_to_4_no-centering_wordrate-rms_chris-log_frequency"]['Pearson_coeff']
GPT2_Semantics_maps = data["GPT-2_L-4_H-768_semantic_context-full_end-epoch-4_split-3_norm-None_temporal-shifting-0_hidden-layer-3cv-alpha_-3_to_4_no-centering_wordrate-rms_chris-log_frequency"]['Pearson_coeff']


diff_imgs_glove_syntax = compute_diff_imgs(Glove_Syntax_maps, Baseline)
diff_imgs_glove_semantic = compute_diff_imgs(Glove_Semantics_maps, Baseline)
diff_imgs_gpt2_syntax = compute_diff_imgs(GPT2_Syntax_maps, Baseline)
diff_imgs_gpt2_semantic = compute_diff_imgs(GPT2_Semantics_maps, Baseline)

imgs_glove = [preference_score(img_sem, img_syn, masker, threshold=3) for (img_sem, img_syn) in zip(diff_imgs_glove_semantic, diff_imgs_glove_syntax)]
imgs_gpt2 = [preference_score(img_sem, img_syn, masker, threshold=3) for (img_sem, img_syn) in zip(diff_imgs_gpt2_semantic, diff_imgs_gpt2_syntax)]


_, _, mask_glove_syntax , fdr_th = group_analysis(diff_imgs_glove_syntax, smoothing_fwhm=6, vmax=None, design_matrix=None, p_val=None, fdr=0.005, mask_img=masker.mask_img_, cluster_threshold=30, height_control='fdr')
_, _, mask_glove_semantics , fdr_th = group_analysis(diff_imgs_glove_semantic, smoothing_fwhm=6, vmax=None, design_matrix=None, p_val=None, fdr=0.005, mask_img=masker.mask_img_, cluster_threshold=30, height_control='fdr')
_, _, mask_gpt2_syntax , fdr_th = group_analysis(diff_imgs_gpt2_syntax, smoothing_fwhm=6, vmax=None, design_matrix=None, p_val=None, fdr=0.005, mask_img=masker.mask_img_, cluster_threshold=30, height_control='fdr')
_, _, mask_gpt2_semantics , fdr_th = group_analysis(diff_imgs_gpt2_semantic, smoothing_fwhm=6, vmax=None, design_matrix=None, p_val=None, fdr=0.005, mask_img=masker.mask_img_, cluster_threshold=30, height_control='fdr')


img_glove = concat_imgs(imgs_glove)
img_gpt2 = concat_imgs(imgs_gpt2)
mask_glove = math_img('img1+img2 > 0', img1=mask_glove_syntax, img2=mask_glove_semantics)
mask_gpt2 = math_img('img1+img2 > 0', img1=mask_gpt2_syntax, img2=mask_gpt2_semantics)


nib.save(img_glove, "/neurospin/unicog/protocols/IRMf/LePetitPrince_Pallier_2018/LePetitPrince/img_glove_threshold-3_corrected_random_ids.nii.gz")
nib.save(mask_glove, "/neurospin/unicog/protocols/IRMf/LePetitPrince_Pallier_2018/LePetitPrince/mask_glove_corrected_random_ids.nii.gz")
nib.save(img_gpt2, "/neurospin/unicog/protocols/IRMf/LePetitPrince_Pallier_2018/LePetitPrince/img_gpt2_threshold-3_corrected_random_ids.nii.gz")
nib.save(mask_gpt2, "/neurospin/unicog/protocols/IRMf/LePetitPrince_Pallier_2018/LePetitPrince/mask_gpt2_corrected_random_ids.nii.gz")

#scp ap263679@nautilus.intra.cea.fr:/neurospin/unicog/protocols/IRMf/LePetitPrince_Pallier_2018/LePetitPrince/img_glove_threshold-3_corrected_random_ids.nii.gz /Users/alexpsq/Code/Parietal/LePetitPrince/derivatives/fMRI/maps/tmp
#scp ap263679@nautilus.intra.cea.fr:/neurospin/unicog/protocols/IRMf/LePetitPrince_Pallier_2018/LePetitPrince/img_gpt2_threshold-3_corrected_random_ids.nii.gz /Users/alexpsq/Code/Parietal/LePetitPrince/derivatives/fMRI/maps/tmp
#scp ap263679@nautilus.intra.cea.fr:/neurospin/unicog/protocols/IRMf/LePetitPrince_Pallier_2018/LePetitPrince/mask_glove_corrected_random_ids.nii.gz /Users/alexpsq/Code/Parietal/LePetitPrince/derivatives/fMRI/maps/tmp
#scp ap263679@nautilus.intra.cea.fr:/neurospin/unicog/protocols/IRMf/LePetitPrince_Pallier_2018/LePetitPrince/mask_gpt2_corrected_random_ids.nii.gz /Users/alexpsq/Code/Parietal/LePetitPrince/derivatives/fMRI/maps/tmp


zmap_glove, effmap_glove, mask_glove_ttest, fdr_th = group_analysis(
            [index_img(img_glove, i) for i in range(img_glove.shape[-1])], 
            smoothing_fwhm=6, 
            design_matrix=None, 
            p_val=None, 
            fdr=0.005, 
            mask_img=math_img('img1*img2', img1=masker.mask_img_, img2=mask_glove),
            cluster_threshold=30, 
            height_control='fdr')
print(fdr_th)


#%%
zmap_gpt2, effmap_gpt2, mask_gpt2_ttest, fdr_th = group_analysis(
            [index_img(img_gpt2, i) for i in range(img_gpt2.shape[-1])], 
            smoothing_fwhm=6, 
            design_matrix=None, 
            p_val=None, 
            fdr=0.005, 
            mask_img=math_img('img1*img2', img1=masker.mask_img_, img2=mask_gpt2),
            cluster_threshold=30, 
            height_control='fdr')
print(fdr_th)



names_sensitivity_ttest = ['glove_ratio_final_ttest_fdr_005', 'gpt2_ratio_final_ttest_fdr_005']
syn_sem_surf_projs_sensitivity_corrected_random_ids_ttest = viz.compute_surf_proj([effmap_glove, effmap_gpt2], zmaps=None, masks=[mask_glove_ttest, mask_gpt2_ttest], ref_img=ref_img, names=names_sensitivity_ttest, categorical_values=None, inflated=False, hemispheres=['left', 'right'], views=['lateral', 'medial'], kind='line', template=None)

hemispheres = ['left', 'right']
views = ['lateral', 'medial']
categorical_values = False
inflated = False
format_figure = 'pdf'
dpi = 300

for i, name in enumerate(names_sensitivity_ttest):
    vmax = 1
    cmap = cmap4
    for h, hemi in enumerate(hemispheres):
        for v, view in enumerate(views):
            figure, axes = viz.create_grid(nb_rows=1, nb_columns=1, row_size_factor=6, overlapping=6, column_size_factor=12)

            #if hemi=='left' and view=='lateral':
            ax = axes[0][0]
            kwargs = viz.set_projection_params(hemi, view, cmap=cmap, 
            inflated=inflated, threshold=1e-15, colorbar=False, symmetric_cbar=False, template=None, figure=figure, ax=ax, vmax=vmax)

            surf_img = syn_sem_surf_projs_sensitivity_corrected_random_ids_ttest[name][f'{hemi}-{view}']
            plotting.plot_surf_stat_map(stat_map=surf_img,**kwargs)
            
            plot_name = f'{name}_{hemi}_{view}'
            print(plot_name)
            format_figure = format_figure
            dpi = dpi
            plt.savefig(os.path.join(saving_folder, f'{plot_name}.{format_figure}'), format=format_figure, dpi=dpi, bbox_inches = 'tight', pad_inches = 0, )
            plt.show()
            plt.close('all')



names_sensitivity = ['glove_ratio_final', 'gpt2_ratio_final']
syn_sem_surf_projs_sensitivity_corrected_random_ids = viz.compute_surf_proj([effmap_glove, effmap_gpt2], zmaps=None, masks=[mask_glove, mask_gpt2], ref_img=ref_img, names=names_sensitivity, categorical_values=None, inflated=False, hemispheres=['left', 'right'], views=['lateral', 'medial'], kind='line', template=None)


for i, name in enumerate(names_sensitivity):
    vmax = 1
    cmap = cmap4
    for h, hemi in enumerate(hemispheres):
        for v, view in enumerate(views):
            figure, axes = viz.create_grid(nb_rows=1, nb_columns=1, row_size_factor=6, overlapping=6, column_size_factor=12)

            #if hemi=='left' and view=='lateral':
            ax = axes[0][0]
            kwargs = viz.set_projection_params(hemi, view, cmap=cmap, 
            inflated=inflated, threshold=1e-15, colorbar=False, symmetric_cbar=False, template=None, figure=figure, ax=ax, vmax=vmax)

            surf_img = syn_sem_surf_projs_sensitivity_corrected_random_ids[name][f'{hemi}-{view}']
            plotting.plot_surf_stat_map(stat_map=surf_img,**kwargs)
            
            plot_name = f'{name}_{hemi}_{view}'
            print(plot_name)
            format_figure = format_figure
            dpi = dpi
            plt.savefig(os.path.join(saving_folder, f'{plot_name}.{format_figure}'), format=format_figure, dpi=dpi, bbox_inches = 'tight', pad_inches = 0, )
            plt.show()
            plt.close('all')
