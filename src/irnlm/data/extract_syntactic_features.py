import os
import gc
from tqdm import tqdm
from joblib import Parallel, delayed

from irnlm.utils import save_pickle, write, check_folder
from irnlm.data.text_tokenizer import tokenize
from irnlm.data.utils import get_possible_morphs, get_possible_pos


def extract_syntax(doc, morphs, pos):
    """Extract number of closing nodes for each words of an input sequence."""
    ncn = []
    morph = []
    pos_ = []
    for sent in doc.sents:
        parsed_string = sent._.parse_string
        # words = sent.text.split(" ")
        for token in sent:
            m = str(morphs.index(str(token.morph))) if str(token.morph) != "" else "0"
            if len(m) == 1:
                m = "00" + m
            elif len(m) == 2:
                m = "0" + m
            morph.append(m)
            p = str(pos.index(token.pos_))
            if len(p) == 1:
                p = "0" + p
            pos_.append(p)
            word = token.text + ")"
            index = parsed_string.find(word)
            l = len(word)
            count = 1
            i = index + l
            while i < len(parsed_string) and parsed_string[i] == ")":
                count += 1
                i += 1
            ncn.append(
                str(min(count, 9))
            )  # we take into account a maximum of 9 closing parenthesis
            parsed_string = parsed_string[i:]
    # print(morphs, ncn)
    return [int("".join(items)) for items in list(zip(ncn, morph, pos_))]


def integral2syntactic(
    path,
    nlp,
    transform_ids,
    language="english",
    convert_numbers=False,
    index=None,
    nblocks=1000,
    normalize=True,
    saving_path="./syntactic_activations.pkl",
    parallel=True,
):
    """Extract syntactic features from the integral text.
    Args:
        - path: list of str (sentences)
        - nlp: Spacy NLP pipelne
        - transform_ids: dict (mapping ids)
        - language: str
        - convert_numbers: bool
        - convert_numbers: bool
        - index: str
        - nblocks: int
        - normalize: bool, whether to use `transform_ids` to change the `ids` found.
        - saving_path: str
    Returns:
        - iterator: list of str (content words)
    """
    check_folder(
        os.path.join(
            os.path.dirname(saving_path),
            "CC100_syntactic_train_split-" + os.path.basename(path).split(".")[1],
        )
    )
    output_path = os.path.join(
        os.path.dirname(saving_path),
        "CC100_syntactic_train_split-" + os.path.basename(path).split(".")[1],
        f"tmp{index}.pkl",
    )
    if not os.path.exists(output_path):
        morphs = get_possible_morphs(nlp)
        pos = get_possible_pos()

        iterator = tokenize(
            path,
            language=language,
            with_punctuation=True,
            convert_numbers=convert_numbers,
        )
        iterator = [item.strip() for item in iterator]
        n = len(iterator)
        if index is not None:
            iterator = iterator[index * n // nblocks : (index + 1) * n // nblocks]
            n = len(iterator)
        else:
            index = ""
        # iterator = [' '.join([word.lower() for word in sent.split(' ')]) for sent in iterator]
        # We group sentences in batches of 100 sentences for computational efficiency
        n_subblocks = 200
        iterator = [
            " ".join(
                iterator[index * n // n_subblocks : (index + 1) * n // n_subblocks]
            )
            for index in range(n_subblocks)
        ]
        n = len(iterator)
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        if parallel:
            activations = Parallel(n_jobs=18)(
                delayed(lambda x: extract_syntax(nlp(x), morphs, pos))(sent)
                for sent in tqdm(iterator, desc="Applying pipeline.", total=n)
                if sent != ""
            )
        else:
            activations = [
                extract_syntax(nlp(sent), morphs, pos)
                for sent in tqdm(iterator, desc="Applying pipeline.", total=n)
                if sent != ""
            ]

        gc.collect()
        # docs = [
        #    nlp(sent)
        #    for sent in tqdm(iterator, desc="Applying pipeline.", total=n)
        #    if sent != ""
        # ]

        # n = len(docs)
        # sentences = [
        #    doc.text.split(" ") for doc in tqdm(docs, desc="Splitting to words.", total=n)
        # ]
        # activations = [
        #    extract_syntax(doc) for doc in tqdm(docs, desc="Extracting syntax.", total=n)
        # ]

        save_pickle(output_path, activations)
    if normalize:
        iterator = []
        for index, activ in tqdm(
            enumerate(activations), total=n, desc="Normalizing values."
        ):
            tmp = []
            for i, value in enumerate(activ):
                if value in transform_ids.keys():
                    tmp.append(
                        transform_ids[value] + 5
                    )  # to leave first indexes to special tokens: ["<pad>", "<s>", "</s>", "<unk>", "<mask>"]
                # else:
                #    print(value, "-", sentences[index][i])
            iterator.append(tmp)
        iterator = [i for l in iterator for i in l]

        return iterator
