# > http://data.statmt.org/cc-100/fr.txt.xz
#### decompress file
# > split --verbose -a 3 -d -b500M data training_data.
#
#
import os
import pickle
from tqdm import tqdm
from typing import Any
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Digits, Whitespace
from tokenizers import Tokenizer, normalizers, pre_tokenizers
from tokenizers.normalizers import NFD, Lowercase, StripAccents

from transformers import PreTrainedTokenizerFast


import benepar
import spacy
from spacy.symbols import ORTH

from irnlm.data.text_tokenizer import tokenize
from irnlm.data.utils import set_nlp_pipeline, get_ids_syntax
from irnlm.data.extract_syntactic_features import integral2syntactic, extract_syntax
from irnlm.data.utils import get_possible_morphs, get_possible_pos


def load_nlp_pipeline(
    language,
    download_dir="/neurospin/unicog/protocols/IRMf/LePetitPrince_Pallier_2018/LePetitPrince/Information-Restrited-NLMs/data",
):
    """ """
    benepar_model = {
        "english": "benepar_en3",
        "french": "benepar_fr2",
    }
    model = {
        "english": "en_core_web_lg",
        "french": "fr_core_news_lg",
    }
    nlp = set_nlp_pipeline(name=model[language])
    special_case = [{ORTH: "hasnt"}]
    nlp.tokenizer.add_special_case("hasnt", special_case)
    benepar.download(
        benepar_model[language],
        download_dir=download_dir,
    )
    spacy.require_cpu()
    nlp.add_pipe(
        "benepar",
        config={"model": os.path.join(download_dir, "models", benepar_model[language])},
    )
    nlp.max_length = 1000000000
    print(nlp.pipe_names)
    return nlp


def read_bpe(
    path: str,
    max_length: int = 512,
    training_data_paths: list = None,
    vocab_size: int = 5000,
    # special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"],
    special_tokens=["<pad>", "<s>", "</s>", "<unk>", "<mask>"],
    model_type="integral",
    language="french",
):
    """Load a pretrained BPE Tokenizer.
    Args:
        - path: str
        - max_length: int
    Returns:
        - tokenizer object
    """
    if model_type == "syntactic":
        tokenizer = TokenizerSyntax(path, language=language)
    else:
        if os.path.exists(f"{path}/bpe-vocab.json"):
            # tokenizer = PreTrainedTokenizerFast(
            #    tokenizer_file=f"{path}/bpe-vocab.json",
            #    bos_token="[CLS]",
            #    eos_token="[SEP]",
            #    unk_token="[UNK]",
            #    sep_token="[SEP]",
            #    pad_token="[PAD]",
            #    cls_token="[CLS]",
            #    mask_token="[MASK]",
            #    padding_side="right",
            # )
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
            # tokenizer.enable_truncation(max_length=max_length)
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
    # special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"],
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
    # tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    tokenizer.normalizer = normalizers.Sequence([NFD(), Lowercase(), StripAccents()])
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence(
        [Digits(individual_digits=True), Whitespace()]
    )
    # tokenizer.post_processor = TemplateProcessing(
    #    single="[CLS] $A [SEP]", special_tokens=[("[CLS]", 1), ("[SEP]", 2)]
    # )
    # tokenizer.post_processor = TemplateProcessing(
    #    single="<s> $A </s>", special_tokens=[("<s>", 1), ("</s>", 2)]
    # )
    # Train the tokenizer
    trainer = BpeTrainer(vocab_size=vocab_size, special_tokens=special_tokens)
    tokenizer.train(files=training_data_paths, trainer=trainer)

    # set tokenizer properties
    tokenizer.__len__ = property(lambda self: self.vocab_size)
    tokenizer.enable_truncation(max_length=max_length)

    # print(tokenizer.encode("le rouge.").ids)
    # print(tokenizer.encode("le rouge."))
    # Save the tokenizer
    tokenizer.save(f"{saving_folder}/bpe-vocab.json")
    return tokenizer


def save_object(filename, data):
    """Save computed examples and features."""
    with open(filename, "wb") as outp:  # Overwrites any existing file.
        pickle.dump(data, outp, pickle.HIGHEST_PROTOCOL)


def load_object(filename):
    """Load computed examples and features."""
    with open(filename, "rb") as inp:  # Overwrites any existing file.
        data = pickle.load(inp)
    return data


class TokenizerSyntax(object):
    def __init__(
        self,
        mapping,
        language="french",
        download_dir="/neurospin/unicog/protocols/IRMf/LePetitPrince_Pallier_2018/LePetitPrince/Information-Restrited-NLMs/data",
    ) -> None:
        super().__init__()
        self.bos_token = "<s>"
        self.eos_token = "</s>"
        self.pad_token = "<pad>"
        self.unk_token = "<unk>"
        self.mask_token = "<mask>"
        self.language = language

        self.nlp = load_nlp_pipeline(language, download_dir=download_dir)
        self.transform_ids = get_ids_syntax(folder=mapping, language=language)
        self.morphs = get_possible_morphs(self.nlp)
        self.pos = get_possible_pos()

    def __call__(self, x, convert_numbers=False) -> Any:
        if x == self.bos_token:
            output = {"input_ids": [1]}
        elif x == self.eos_token:
            output = {"input_ids": [2]}
        elif x == self.pad_token:
            output = {"input_ids": [0]}
        elif x == self.unk_token:
            output = {"input_ids": [3]}
        elif x == self.mask_token:
            output = {"input_ids": [4]}
        else:
            output = self.encode(x, convert_numbers=convert_numbers)
        return output

    def encode(self, path, convert_numbers=False):
        iterator = tokenize(
            path,
            language=self.language,
            with_punctuation=True,
            convert_numbers=convert_numbers,
        )
        iterator = [item.strip() for item in iterator]
        n = len(iterator)
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        docs = [self.nlp(x) for x in iterator]
        activations = [extract_syntax(doc, self.morphs, self.pos) for doc in docs]
        output = []
        for index, activ in tqdm(
            enumerate(activations), total=n, desc="Normalizing values."
        ):
            tmp = []
            for i, value in enumerate(activ):
                if value in self.transform_ids.keys():
                    tmp.append(
                        self.transform_ids[value] + 5
                    )  # to leave first indexes to special tokens: ["<pad>", "<s>", "</s>", "<unk>", "<mask>"]
                # else:
                #    print(value, "-", sentences[index][i])
            output.append(tmp)
        output = {"input_ids": [i for l in output for i in l]}
        return output
