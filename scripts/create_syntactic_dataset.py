import argparse

import benepar
from spacy.symbols import ORTH

from irnlm.data.utils import set_nlp_pipeline, get_ids_syntax
from irnlm.data.extract_syntactic_features import integral2syntactic
from irnlm.utils import save_pickle, write


nlp = set_nlp_pipeline()
special_case = [{ORTH: "hasnt"}]
nlp.tokenizer.add_special_case("hasnt", special_case)
benepar.download('benepar_en3')
nlp.add_pipe('benepar', config={'model': 'benepar_en3'})
print(nlp.pipe_names)


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Extract syntactic features (POS/Morph/NCN) from the integral text and convert them to ids.')
    parser.add_argument("--text", type=str)
    parser.add_argument("--saving_path", type=str, default="./syntactic_features.pkl")

    args = parser.parse_args()
    path_integral_text = args.text
    
    transform_ids = get_ids_syntax()

    syntactic_features = integral2syntactic(path_integral_text, nlp, transform_ids=transform_ids)

    save_pickle(args.saving_path, syntactic_features)
    write(args.saving_path.replace('.pkl', '.txt'), ' '.join([str(i) for i in syntactic_features]))
