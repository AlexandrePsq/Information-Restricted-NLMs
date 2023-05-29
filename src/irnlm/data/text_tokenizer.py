import os
import re
import inflect
from tqdm import tqdm

import numpy as np
import scipy.io.wavfile as wave
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize

special_words = {
    "english": {
        "grown-ups": "grownups",
        "grown-up": "grownup",
        "hasn't": "hasnt",
        "hasn‘t": "hasnt",
        "redfaced": "red faced",
    },
    "french": {},
}


def tokenize(
    path,
    language,
    with_punctuation=True,
    convert_numbers=True,
    max_sent_size=512,
    margin=0.6,
):
    """Tokenize a text into sentences.
    Optionnaly preprocess it.
    Arguments:
        - path: (str) path or text
        - language: (str)
    Returns:
        - iterator: sentence iterator (with/without punctuation)
    """
    punctuation = [
        "'",
        ",",
        ";",
        ":",
        "/",
        "-",
        '"',
        "‘",
        "’",
        "(",
        ")",
        "{",
        "}",
        "[",
        "]",
        "`",
        "“",
        "”",
        "—",
        ".",
        "!",
        "?",
        "«",
        "»",
    ]
    if os.path.exists(path):
        try:
            path = open(path, "r", encoding="utf8").read()
        except:
            print(f"Cannot read {path}")
    text = preprocess(path, special_words, language, convert_numbers)
    iterator = [
        item.strip() for item in tqdm(text.split("\n")[:-1])
    ]  # vocab words not lowered
    if not with_punctuation:
        for item in punctuation:
            iterator = [sentence.replace(item, "") for sentence in iterator]
    iterator = [re.sub(" +", " ", sentence).strip() for sentence in iterator]
    iterator = [item for item in iterator if item != ""]
    iterator = [
        item
        if len(item.split(" ")) <= max_sent_size * margin
        else " ".join(item.split(" ")[: int(max_sent_size * margin) - 1])
        + " ."  # benepar tokenizer is buggy with sentences
        # containing more than 512 tokens...
        # so you have to consider an error margin
        # for the maximum number of tokens in each sentence.
        for item in iterator
    ]
    return iterator


def preprocess(text, special_words, language, convert_numbers=False):
    """Prepare text for tokenization into sentences.
    Replace words in text by the ones by which they have been replaced in the
    textgrid files. Then replace all numbers by their written english version.
    We then add a space on each side of every punctuation symbol, paying attention
    to '...'.
    Arguments:
        - text: (str) text to preprocess
        - special_words: (dict) special words and words by which to replace them
        - language: (str)
    """
    text = text.replace("\n", " ")
    text = text.replace("<unk>", "unk")
    for word in special_words[language].keys():
        text = text.replace(word, special_words[language][word])
    if convert_numbers:
        transf = inflect.engine()
        numbers = re.findall("\d+", text)
        for number in tqdm(numbers, desc="Converting numbers", total=len(numbers)):
            text = text.replace(number, transf.number_to_words(number))
    if language == "french":
        punctuation = [
            ",",
            ";",
            ":",
            "/",
            "-",
            '"',
            "‘",
            "’",
            "(",
            ")",
            "{",
            "}",
            "[",
            "]",
            "`",
            "“",
            "”",
            "—",
            "«",
            "»",
            "'",
        ]
        text = text.replace("'", "' ")
    elif language == "english":
        punctuation = [
            "'",
            ",",
            ";",
            ":",
            "/",
            "-",
            '"',
            "‘",
            "’",
            "(",
            ")",
            "{",
            "}",
            "[",
            "]",
            "`",
            "“",
            "”",
            "—",
        ]
    eos_punctuation = [".", "!", "?"]
    for item in punctuation:
        text = text.replace(item, " " + item + " ")
    text = text.replace("...", "<3 points>")
    for item in eos_punctuation:
        text = text.replace(item, " " + item + "\n")
    text = text.replace("<3 points>", " ...\n")
    for item in eos_punctuation + ["..."]:
        text = text.replace(item + "\n" + " " + '"', item + " " + '"' + "\n")
        text = text.replace(item + "\n" + " " + "»", item + " " + "»" + "\n")
        text = text.replace(item + "\n" + " " + "”", item + " " + "”" + "\n")
        text = text.replace(item + "\n" + " " + "’", item + " " + "’" + "\n")
    text = re.sub(" +", " ", text)
    return text


def rms_tokenizer(path_to_audio, slice_period):
    # slice_period is in s
    # wave_file = wave.open(path, mode='r')
    # rate = wave_file.getframerate()
    # n_frames = wave_file.getnframes()   # Number of frames.
    # slice_length = int(slice_period * rate)
    [rate, data] = wave.read(path_to_audio)
    slice_length = int(slice_period * rate)
    data_list = [
        np.array(
            data[index * slice_length : index * slice_length + slice_length],
            dtype=np.float64,
        )
        for index in range(len(data) // slice_length)
    ]
    # Read audio data.
    # data = np.frombuffer(wave_file.readframes(n_frames), dtype=np.int16)
    # data_list = [data[index*slice_length: index*slice_length + slice_length] for index in range(n_frames//slice_length)]
    return data_list, rate, len(data), slice_length
