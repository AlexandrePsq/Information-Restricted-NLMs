import argparse

from irnlm.models.glove.glove import extract_features as glove_int_extractor
from irnlm.models.glove.glove import load_model_and_tokenizer as load_glove_int
from irnlm.models.glove_semantic.glove import extract_features as glove_sem_extractor
from irnlm.models.glove_semantic.glove import load_model_and_tokenizer as load_glove_sem
from irnlm.models.glove_syntactic.glove import extract_features as glove_syn_extractor
from irnlm.models.glove_syntactic.glove import (
    load_model_and_tokenizer as load_glove_syn,
)
from irnlm.models.gpt2.extract_features_gpt2_integral import (
    extract_features as gpt2_int_extractor,
)
from irnlm.models.gpt2.extract_features_gpt2_integral import (
    load_model_and_tokenizer as load_gpt2,
)
from irnlm.models.gpt2.extract_features_gpt2_semantic import (
    extract_features as gpt2_sem_extractor,
)
from irnlm.models.gpt2.extract_features_gpt2_syntactic import (
    extract_features as gpt2_syn_extractor,
)
from irnlm.utils import filter_args

extractor_dict = {
    "gpt2": {
        "integral": gpt2_int_extractor,
        "semantic": gpt2_sem_extractor,
        "syntactic": gpt2_syn_extractor,
    },
    "glove": {
        "integral": glove_int_extractor,
        "semantic": glove_sem_extractor,
        "syntactic": glove_syn_extractor,
    },
}
model_tokenizer_dict = {
    "gpt2": {
        "integral": lambda x: load_gpt2(trained_model=x, model_type="integral"),
        "semantic": lambda x: load_gpt2(trained_model=x, model_type="semantic"),
        "syntactic": lambda x: load_gpt2(trained_model=x, model_type="syntactic"),
    },
    "glove": {
        "integral": load_glove_int,
        "semantic": load_glove_sem,
        "syntactic": load_glove_syn,
    },
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="""Tokenize a Dataset using the right tokenizer (integral, semantic, syntactic)."""
    )
    parser.add_argument("--path", type=str)
    parser.add_argument("--model", type=str, default="glove")  #'gpt2'
    parser.add_argument(
        "--type", type=str, default="integral"
    )  #'semantic', 'syntactic'
    parser.add_argument("--context_size", type=int, default=507)
    parser.add_argument("--max_seq_length", type=int, default=512)
    parser.add_argument("--config", type=str)
    parser.add_argument("--bsz", type=int, default=32)
    parser.add_argument("--language", type=str, default="english")
    parser.add_argument("--saving_path", type=str, default="derivatives/features.csv")

    args = parser.parse_args()
    extractor_func = extractor_dict[args.model][args.type]
    model, tokenizer = model_tokenizer_dict[args.model][args.type](args.config)

    NUM_HIDDEN_LAYERS = model.config.num_hidden_layers if args.model == "gpt2" else None
    FEATURE_COUNT = (
        model.config.hidden_size
        if args.model == "gpt2"
        else len(model[list(model.keys())[0]])
    )

    kwargs = {
        "path": args.path,
        "model": model,
        "tokenizer": tokenizer,
        "context_size": args.context_size,
        "max_seq_length": args.max_seq_length,
        "FEATURE_COUNT": FEATURE_COUNT,
        "NUM_HIDDEN_LAYERS": NUM_HIDDEN_LAYERS,
        "bsz": args.bsz,
        "language": args.language,
    }
    kwargs = filter_args(extractor_func, kwargs)
    features = extractor_func(**kwargs)

    features.to_csv(args.saving_path)
