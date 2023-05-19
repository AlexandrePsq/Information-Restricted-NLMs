""" Code to fine-tune hugging-face implementation of GPT2 model.
https://huggingface.co/
"""
import warnings

warnings.simplefilter(action="ignore")

import os
import gc
import glob
import logging
import argparse
import pandas as pd

from transformers import GPT2LMHeadModel, GPT2Tokenizer

from irnlm.models.tokenizer import read_bpe
from irnlm.models.utils import set_seed, get_device
from irnlm.models.gpt2.processors import ModelProcessor
from irnlm.utils import read_yaml, save_yaml, check_folder
from irnlm.models.gpt2.language_modeling import LMDataset, LMProcessor

########################################################################################################
# ------------------------------------------- FINE - TUNING -------------------------------------------#
########################################################################################################

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fine-tune a GPT2 model for a specific NLP task."
    )
    parser.add_argument(
        "--yaml_file",
        type=str,
        help="""Path to the yaml file containing additional information on how 
                                                        the dataset is structured.""",
    )
    args = parser.parse_args()

    # Fetch parameters
    parameters = read_yaml(args.yaml_file)
    check_folder(parameters["output_dir"])
    nb_splits = parameters["nb_splits"]
    save_yaml(parameters, os.path.join(parameters["output_dir"], "config_eval.yml"))
    logging.basicConfig(
        filename=os.path.join(parameters["output_dir"], parameters["log_file"]),
        filemode="w+",
        level=logging.INFO,
    )
    logging.info("Parameters fetched.")

    logging.info("Set and retrieve the device on which to run...")
    device = get_device()
    task = parameters["task"].lower()
    logging.info("\tDone.")

    logging.info("Instanciating data processor and data paths...")
    processor = LMProcessor(
        train_paths=parameters["train_paths"],
        dev_paths=parameters["dev_paths"],
        test_paths=parameters["test_paths"],
        max_seq_length=parameters["max_length"],
        device=device,
        output_dir=parameters["output_dir"],
        dataset_dir=parameters["dataset_dir"],
        context_size=parameters["context_size"],
        n_splits=None,
    )
    logging.info("\tDone.")

    dev_data_paths = processor.get_data("dev") if parameters["do_validation"] else []
    test_data_paths = processor.get_data("test") if parameters["do_test"] else []

    # Setting environment for the tokenizer not to interefere with future parallelisation
    logging.info("Fetching Tokenizer...")
    if parameters["pretrained_tokenizer"] is not None:
        tokenizer = GPT2Tokenizer.from_pretrained(parameters["pretrained_tokenizer"])
    else:
        tokenizer = read_bpe(
            path=parameters["tokenizer_path"],
            max_length=parameters["max_length"],  # max_length
            training_data_paths=parameters["tokenizer_training_data_paths"],
            vocab_size=parameters["vocab_size"],
            special_tokens=parameters["tokenizer_special_tokens"],
        )

    processor.set_tokenizer(tokenizer)
    # Setting environment for the tokenizer not to interefere with future parallelisation
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    logging.info("\tDone.")

    logging.info("Evaluating the model.")
    gc.collect()
    model_processor = ModelProcessor(
        None,
        None,
        tokenizer,
        None,
        device,
        parameters["metric_name"],
        None,
        parameters["use_output_mask"],
        context_size=parameters["context_size"],
    )

    try:
        path = os.path.join(parameters["output_dir"], parameters["model"])
        print(f"Using model saved at: {path}...")
        logging.info(f"Using model saved at: {path}...")
        logging.info("Setting seed for reproductibility...")
        set_seed(parameters["seed"])
        model_processor.model = GPT2LMHeadModel.from_pretrained(
            path,
            output_attentions=parameters[
                "output_attentions"
            ],  # Whether the model returns attentions weights.
            output_hidden_states=parameters[
                "output_hidden_states"
            ],  # Whether the model returns all hidden-states.
        )
        if parameters["device_map"] is not None:
            device_map = parameters["device_map"]
            model_processor.model.parallelize(device_map)
            print(f"Using device_map: {device_map}")
        else:
            model_processor.model.to(device)

        accuracy, loss = None, None

        logging.info("Validation: ")
        val_loss, val_time = model_processor.evaluate(
            processor, dev_data_paths, "dev", parameters
        )
        stats = {
            "Valid. Loss": val_loss,
            "Valid. Time": val_time,
        }

        logging.info("Test: ")
        test_loss, test_time = model_processor.evaluate(
            processor, test_data_paths, "dev", parameters
        )
        stats["Test. Loss"] = test_loss
        stats["Test. Time"] = test_time

        df = pd.DataFrame(data=stats)
        df.to_csv(
            os.path.join(
                parameters["output_dir"], f"eval_stats_{parameters['model']}.csv"
            ),
            index=False,
        )
        logging.info("\tDone.")

    except KeyboardInterrupt:
        print("-" * 89)
        print("Exiting from training early")
