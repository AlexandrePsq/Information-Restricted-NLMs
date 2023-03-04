from irnlm.utils import check_folder
from fmri_encoder.data import fetch_masker
from irnlm.contrasts import plot_brain_fit


if __name__=='__main__':

    ###
    # Replicating Fig6: Supra-lexical vs Lexical
    ###
    saving_folder = 'derivatives/Fig6'
    check_folder(saving_folder)
    masker = fetch_masker('derivatives/masker', None, **{'detrend': False, 'standardize': False})

    # Semantics
    name0 = "GPT-2_L-4_H-768_semantic_context-full_end-epoch-4_split-3_norm-None_temporal-shifting-0_{}_hidden-layer-3cv-alpha_-3_to_4_no-centering_wordrate-rms_chris-log_frequency"
    name1 = "GloVe_Semanticcv-alpha_-3_to_4wordrate-rms_chris-log_frequency_{}"
    effmaps_dict, zmaps_dict = plot_brain_fit(
        name0, name1, vmax=0.04, cmap='cold_hot', surname='GPT-2_Semantic-vs-GloVe_Semantic', 
        masker=masker, height_control='fdr', alpha=0.005, saving_folder=saving_folder)
    
    # Syntax
    name0 = "GPT-2_L-4_H-768_Syntax_context-full_end-epoch-4_split-3_norm-None_temporal-shifting-0_{}_hidden-layer-3cv-alpha_-3_to_4_no-centering_wordrate-rms_chris-log_frequency"
    name1 = "GloVe_Syntax_fullcv-alpha_-3_to_4wordrate-rms_chris-log_frequency_{}"
    effmaps_dict, zmaps_dict = plot_brain_fit(
        name0, name1, vmax=0.04, cmap='cold_hot', surname='GPT-2_Syntax-vs-GloVe_Syntax', 
        masker=masker, height_control='fdr', alpha=0.005, saving_folder=saving_folder)
    
    # Integral
    name0 = "GPT-2_L-4_H-768_default_tokenizer_full_end-epoch-4_split-3_norm-None_temporal-shifting-0_{}_hidden-layer-3cv-alpha_-3_to_4_no-centering_wordrate-rms_chris-log_frequency"
    name1 = "GloVecv-alpha_-3_to_4wordrate-rms_chris-log_frequency_{}"
    effmaps_dict, zmaps_dict = plot_brain_fit(
        name0, name1, vmax=0.04, cmap='cold_hot', surname='GPT-2_full-vs-GloVe_full', 
        masker=masker, height_control='fdr', alpha=0.005, saving_folder=saving_folder)
    
