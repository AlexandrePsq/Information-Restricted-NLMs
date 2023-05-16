import argparse

import benepar
from spacy.symbols import ORTH

from irnlm.data.utils import set_nlp_pipeline, get_ids_syntax
from irnlm.data.extract_syntactic_features import integral2syntactic
from irnlm.utils import save_pickle, write


def load_nlp_pipeline(language):
    """
    """
    benepar_model = {
        'english': "benepar_en3",
        'french': "benepar_fr",
    }
    model = {
        'english': "en_core_web_lg",
        'french': "fr_core_news_lg",
    }
    nlp = set_nlp_pipeline(name=model[language])
    special_case = [{ORTH: "hasnt"}]
    nlp.tokenizer.add_special_case("hasnt", special_case)
    benepar.download(benepar_model[language])
    nlp.add_pipe("benepar", config={"model": benepar_model[language])})
    print(nlp.pipe_names)
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract syntactic features (POS/Morph/NCN) from the integral text and convert them to ids."
    )
    parser.add_argument("--text", type=str)
    parser.add_argument("--index", type=int, default=None)
    parser.add_argument("--nblocks", type=int, default=1000)
    parser.add_argument("--language", type=str, default="english")
    parser.add_argument("--normalize", type=bool, default=False, action="store_true")
    parser.add_argument("--saving_path", type=str, default="./syntactic_features.pkl")

    args = parser.parse_args()
    path_integral_text = args.text

    nlp = load_nlp_pipeline(args.language)

    if args.normalize:
        transform_ids = get_ids_syntax(args.language)
    else:
        transform_ids = None

    syntactic_features = integral2syntactic(
        path_integral_text,
        nlp,
        transform_ids=transform_ids,
        language=args.language,
        index=args.index,
        nblocks=args.nblocks,
        normalize=args.normalize,
        saving_path=args.saving_path
    )

    if args.normalize:
        save_pickle(args.saving_path, syntactic_features)
    # write(args.saving_path.replace('.pkl', '.txt'), ' '.join([str(i) for i in syntactic_features]))
