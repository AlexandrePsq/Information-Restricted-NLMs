# > http://data.statmt.org/cc-100/fr.txt.xz
#### decompress file 
# > split --verbose -a 3 -d -b500M data training_data.
#
#
import os
import pathlib
import logging
import torch
import pickle
import pandas as pd
from torch.utils.data import DataLoader, Dataset
import torch
from tqdm import tqdm
from functools import partial
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.processors import TemplateProcessing
from tokenizers.pre_tokenizers import Digits, Whitespace
from tokenizers import Tokenizer, normalizers, pre_tokenizers
from tokenizers.normalizers import NFD, Lowercase, StripAccents

from transformers import PreTrainedTokenizerFast


logging.basicConfig(level=logging.INFO)


        
def read_bpe(
    path: str, 
    max_length: int = 512, 
    training_data_paths: list = None,
    vocab_size: int = 5000,
    #special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"],
    special_tokens=["<pad>", "<s>", "</s>", "<unk>", "<mask>"],
    ):
    """Load a pretrained BPE Tokenizer.
    Args:
        - path: str
        - max_length: int
    Returns:
        - tokenizer object
    """
    if os.path.exists(f"{path}/bpe-vocab.json"):
        logging.info(f"Loading tokenizer from {path}/bpe-vocab.json")
        #tokenizer = PreTrainedTokenizerFast(
        #    tokenizer_file=f"{path}/bpe-vocab.json",
        #    bos_token="[CLS]",
        #    eos_token="[SEP]",
        #    unk_token="[UNK]",
        #    sep_token="[SEP]",
        #    pad_token="[PAD]",
        #    cls_token="[CLS]",
        #    mask_token="[MASK]",
        #    padding_side="right",
        #)
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_file=f"{path}/bpe-vocab.json",
            bos_token="<s>",
            eos_token="</s>",
            unk_token="<unk>",
            sep_token="</s>",
            pad_token="<pad>",
            cls_token="<s>",
            mask_token="<mask>",
            padding_side="right",
        )
        # add length property to tokenizer object
        tokenizer.__len__ = property(lambda self: self.vocab_size)
        #tokenizer.enable_truncation(max_length=max_length)
    else:
        tokenizer = train_bpe_tokenizer(
            training_data_paths,
            path,
            vocab_size=vocab_size,
            max_length=max_length,
            special_tokens=special_tokens,
        )
    return tokenizer

def train_bpe_tokenizer(
    training_data_paths: list,
    saving_folder: str,
    vocab_size: int = 50000,
    max_length: int = 512,
    #special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"],
    special_tokens=["<pad>", "<s>", "</s>", "<unk>", "<mask>"],
    ):
    """Train a BPE tokenizer.
    Args:
        - training_data_paths: list of str
        - saving_folder: str
        - vocab_size: int
        - max_length: int
        - special_tokens: list of str
    Returns:
        - tokenizer object
    """
    # Instantiate the tokenizer
    #tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    tokenizer.normalizer = normalizers.Sequence([NFD(), Lowercase(), StripAccents()])
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence(
        [Digits(individual_digits=True), Whitespace()]
    )
    #tokenizer.post_processor = TemplateProcessing(
    #    single="[CLS] $A [SEP]", special_tokens=[("[CLS]", 1), ("[SEP]", 2)]
    #)
    #tokenizer.post_processor = TemplateProcessing(
    #    single="<s> $A </s>", special_tokens=[("<s>", 1), ("</s>", 2)]
    #)
    # Train the tokenizer
    trainer = BpeTrainer(vocab_size=vocab_size, special_tokens=special_tokens)
    tokenizer.train(files=training_data_paths, trainer=trainer)

    # set tokenizer properties
    tokenizer.__len__ = property(lambda self: self.vocab_size)
    tokenizer.enable_truncation(max_length=max_length)

    print(tokenizer.encode("le rouge.").ids)
    print(tokenizer.encode("le rouge."))
    # Save the tokenizer
    tokenizer.save(f"{saving_folder}/bpe-vocab.json")
    return tokenizer


def save_object(filename, data):
    """Save computed examples and features.
    """
    with open(filename, 'wb') as outp:  # Overwrites any existing file.
        pickle.dump(data, outp, pickle.HIGHEST_PROTOCOL)


def load_object(filename):
    """Load computed examples and features.
    """
    with open(filename, 'rb') as inp:  # Overwrites any existing file.
        data = pickle.load(inp)
    return data