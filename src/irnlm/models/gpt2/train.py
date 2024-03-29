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

from transformers import (
    AdamW,
    GPT2Config,
    get_linear_schedule_with_warmup,
    GPT2Tokenizer,
    GPT2Config,
)

from irnlm.models.gpt2.modeling_hacked_gpt2_integral import GPT2LMHeadModel
from irnlm.models.gpt2.modeling_hacked_gpt2_semantic import (
    GPT2LMHeadModel as GPT2LMHeadModelSemantic,
)
from irnlm.models.gpt2.modeling_hacked_gpt2_semantic_no_relative_pos import (
    GPT2LMHeadModel as GPT2LMHeadModelSemanticNoRelativePos,
)
from irnlm.models.gpt2.language_modeling import LMProcessor
from irnlm.utils import (
    read_yaml,
    check_folder,
    save_yaml,
)
from irnlm.models.utils import (
    set_seed,
    get_device,
    load_last_checkpoint,
    save_checkpoint,
)
from irnlm.models.gpt2.processors import ModelProcessor
from irnlm.models.tokenizer import read_bpe


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
    save_yaml(parameters, os.path.join(parameters["output_dir"], "config.yml"))
    log_path = os.path.join(parameters["output_dir"], parameters["log_file"])
    logging.basicConfig(filename=log_path, filemode="w+", level=logging.INFO)
    logging.info("Parameters fetched.")

    logging.info("Setting seed for reproductibility...")
    set_seed(parameters["seed"])
    logging.info("\tDone.")

    logging.info("Set and retrieve the device on which to run...")
    device = get_device()
    task = parameters["task"].lower()
    logging.info("\tDone.")

    logging.info("Instanciating dataset and data processor...")
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
    logging.info("\tDone.")

    logging.info(f"Fetching GPT-2 model for the task: {parameters['task']}...")
    if task in ["language-modeling"]:
        if parameters["start_from_scratch"]:
            params = read_yaml(parameters["config_path"])
            params["layer_norm_epsilon"] = float(params["layer_norm_epsilon"])
            if parameters["model_type"] in ["integral", "syntactic"]:
                model = GPT2LMHeadModel(GPT2Config(**params))
            elif parameters["model_type"] == "semantic":
                model = GPT2LMHeadModelSemantic(GPT2Config(**params))
            elif parameters["model_type"] == "semantic_no_relative_pos":
                model = GPT2LMHeadModelSemanticNoRelativePos(GPT2Config(**params))
        else:
            if parameters["model_type"] in ["integral", "syntactic"]:
                model = GPT2LMHeadModel.from_pretrained(
                    parameters["pretrained_model"],
                    output_attentions=parameters[
                        "output_attentions"
                    ],  # Whether the model returns attentions weights.
                    output_hidden_states=parameters[
                        "output_hidden_states"
                    ],  # Whether the model returns all hidden-states.
                )
            elif parameters["model_type"] == "semantic":
                model = GPT2LMHeadModelSemantic.from_pretrained(
                    parameters["pretrained_model"],
                    output_attentions=parameters[
                        "output_attentions"
                    ],  # Whether the model returns attentions weights.
                    output_hidden_states=parameters[
                        "output_hidden_states"
                    ],  # Whether the model returns all hidden-states.
                )
            elif parameters["model_type"] == "semantic_no_relative_pos":
                model = GPT2LMHeadModelSemanticNoRelativePos.from_pretrained(
                    parameters["pretrained_model"],
                    output_attentions=parameters[
                        "output_attentions"
                    ],  # Whether the model returns attentions weights.
                    output_hidden_states=parameters[
                        "output_hidden_states"
                    ],  # Whether the model returns all hidden-states.
                )

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
            model_type=parameters["model_type"],
            language=parameters["language"],
        )

    processor.set_tokenizer(tokenizer)

    model, start_at_dataloader = load_last_checkpoint(parameters, model=model)

    if parameters["device_map"] is not None:
        device_map = parameters["device_map"]
        model.parallelize(device_map)
        print(f"Using device_map: {device_map}")
    else:
        model.to(device)

    # Setting environment for the tokenizer not to interefere with future parallelisation
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    logging.info("\tDone.")

    logging.info("Get input data...")
    train_data_paths = processor.get_data("train") if parameters["do_train"] else None
    dev_data_paths = processor.get_data("dev") if parameters["do_validation"] else None
    test_data_paths = processor.get_data("test") if parameters["do_test"] else None
    logging.info("\tDone.")

    logging.info("Creating optimizer and learning rate scheduler...")
    optimizer = AdamW(
        model.parameters(),
        lr=float(parameters["learning_rate"]),
        eps=float(parameters["adam_epsilon"]),
    )
    if parameters["context_size"] == 0:
        nb_steps = sum(
            [len(processor.load_object(path)) // 2 for path in train_data_paths]
        )
    else:
        nb_steps = sum(
            [
                len(processor.load_object(path)) // parameters["context_size"]
                for path in train_data_paths
            ]
        )
    total_steps = (
        nb_steps * parameters["nb_epochs"]
    )  # Total number of training steps is [nb steps] x [nb epochs].
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=parameters["num_warmup_steps"],
        num_training_steps=total_steps,
    )
    logging.info("\tDone.")

    logging.info("Fine-tuning the model.")
    gc.collect()
    model_processor = ModelProcessor(
        model,
        optimizer,
        tokenizer,
        scheduler,
        device,
        parameters["metric_name"],
        parameters["nb_epochs"],
        parameters["use_output_mask"],
        context_size=parameters["context_size"],
        nb_steps=nb_steps,
    )

    try:
        if parameters["do_train"] or parameters["do_validation"]:
            training_stats = model_processor.train(
                processor,
                train_data_paths,
                dev_data_paths,
                parameters["output_dir"],
                parameters=parameters,
                start_at_dataloader=start_at_dataloader,
            )

            logging.info(
                "Saving fine-tuned model to {}...".format(
                    os.path.join(parameters["output_dir"], "fine_tuned")
                )
            )
            name = (
                f"started_at_{parameters['init_checkpoints']}_fine_tuned"
                if parameters["init_checkpoints"] > 0
                else "fine_tuned"
            )
            save_checkpoint(
                model_processor.model, tokenizer, parameters["output_dir"], name
            )
            logging.info("\tDone.")

        else:
            path = sorted(
                glob.glob(os.path.join(parameters["output_dir"], "fine_tuned*"))
            )[-1]
            print(f"Using model saved at: {path}...")
            if parameters["model_type"] in ["integral", "syntactic"]:
                model_processor.model = GPT2LMHeadModel.from_pretrained(
                    path,
                    output_attentions=parameters[
                        "output_attentions"
                    ],  # Whether the model returns attentions weights.
                    output_hidden_states=parameters[
                        "output_hidden_states"
                    ],  # Whether the model returns all hidden-states.
                )
            elif parameters["model_type"] == "semantic":
                model_processor.model = GPT2LMHeadModelSemantic.from_pretrained(
                    path,
                    output_attentions=parameters[
                        "output_attentions"
                    ],  # Whether the model returns attentions weights.
                    output_hidden_states=parameters[
                        "output_hidden_states"
                    ],  # Whether the model returns all hidden-states.
                )
            model_processor.model.to(device)
            training_stats = pd.read_csv(
                os.path.join(parameters["output_dir"], "training_stats.csv")
            )

    except KeyboardInterrupt:
        print("-" * 89)
        training_stats = pd.read_csv(
            os.path.join(parameters["output_dir"], "training_stats.csv")
        )
        print("Exiting from training early")

    # logging.info("Validation reports: ")
    # for epoch, stat in training_stats.iterrows():
    #    logging.info(stat['report'])
    test_accuracy, test_loss = None, None

    if parameters["do_test"]:
        logging.info("Evaluation report: ")
        test_loss, test_time = model_processor.evaluate(
            processor, test_data_paths, "test", parameters
        )
        testing_stats = [
            {
                "Test. Loss": test_loss,
                "Test Time": test_time,
            }
        ]
        df = pd.DataFrame(data=testing_stats)
        df.to_csv(
            os.path.join(parameters["output_dir"], "testing_stats.csv"), index=False
        )
        # logging.info(df['report'].iloc[0])
    logging.info("\tDone.")
