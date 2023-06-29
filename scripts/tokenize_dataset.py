import os
import argparse

from transformers import GPT2Tokenizer

from irnlm.utils import read_yaml, check_folder, save_yaml, save_pickle, get_progress

from irnlm.models.gpt2.modeling_hacked_gpt2_integral import GPT2LMHeadModel
from irnlm.models.gpt2.modeling_hacked_gpt2_semantic import (
    GPT2LMHeadModel as GPT2LMHeadModelSemantic,
)
from irnlm.models.gpt2.modeling_hacked_gpt2_semantic_no_relative_pos import (
    GPT2LMHeadModel as GPT2LMHeadModelSemanticNoRelativePos,
)
from irnlm.models.gpt2.language_modeling import LMProcessor
from irnlm.models.utils import set_seed, get_device
from irnlm.models.gpt2.processors import ModelProcessor
from irnlm.models.tokenizer import read_bpe

os.environ["TOKENIZERS_PARALLELISM"] = "true"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="""Tokenize a Dataset using the right tokenizer (integral, semantic, syntactic)."""
    )
    parser.add_argument("--yaml_file", type=str)

    args = parser.parse_args()

    # Fetch parameters
    parameters = read_yaml(args.yaml_file)
    check_folder(parameters["output_dir"])
    nb_splits = parameters["nb_splits"]
    save_yaml(parameters, os.path.join(parameters["output_dir"], "config.yml"))
    log_path = os.path.join(parameters["output_dir"], parameters["log_file"])

    set_seed(parameters["seed"])

    device = get_device()
    task = parameters["task"].lower()

    if task in ["language-modeling"]:
        processor = LMProcessor(
            train_paths=parameters["train_paths"],
            dev_paths=parameters["dev_paths"],
            test_paths=parameters["test_paths"],
            max_seq_length=parameters["max_length"],
            device=device,
            output_dir=parameters["output_dir"],
            dataset_dir=parameters["dataset_dir"],
            context_size=parameters["context_size"],
            n_splits=nb_splits,
        )

    if parameters["pretrained_tokenizer"] is not None:
        tokenizer = GPT2Tokenizer.from_pretrained(parameters["pretrained_tokenizer"])
    else:
        tokenizer = read_bpe(
            path=parameters["tokenizer_path"],
            max_length=parameters["max_length"],  # max_length
            training_data_paths=parameters["tokenizer_training_data_paths"],
            vocab_size=parameters["vocab_size"],
            special_tokens=parameters["tokenizer_special_tokens"],
            model_type=parameters["model_type"],
            language=parameters["language"],
        )

    processor.set_tokenizer(tokenizer)
    train_data_paths = processor.get_data("train") if parameters["do_train"] else None
    dev_data_paths = processor.get_data("dev") if parameters["do_validation"] else None
    test_data_paths = processor.get_data("test") if parameters["do_test"] else None

    n = len(train_data_paths) + len(dev_data_paths) + len(test_data_paths)

    with get_progress(transient=True) as progress:
        task = progress.add_task(f"Tokenizing", total=n)
        for path in [*train_data_paths, *dev_data_paths, *test_data_paths]:
            iterator = open(path, "r").read()
            try:
                data = tokenizer(iterator)["input_ids"]
            except:
                data = tokenizer.encode(iterator)

            # Creating name
            saving_folder = os.path.dirname(path)
            name = os.path.basename(path).split(".")[0]

            save_pickle(os.path.join(saving_folder, name + ".pkl"), data)
            progress.update(task, advance=1)
