# Information-Restrited-NLMs
Code of the paper: "Information-Restricted Neural Language Models Reveal Different Brain Regions' Sensitivity to Semantics, Syntax and Context"

# Installation & Set up

First, install the requirements.

```shell
git clone git@github.com:AlexandrePsq/Information-Restrited-NLMs.git
cd Information-Restrited-NLMs
pip install -r requirements.txt
pip install -e .
python -m spacy download en_core_web_lg-3.0.0 --direct # this is the version that was used
```

# Loading data

## French Training data: the Integral Dataset

The Integral Dataset is derived from the CC100-French Dataset.
It was downloaded from [CC100-French Dataset](http://data.statmt.org/cc-100/fr.txt.xz).

As we used 5Gb of data for the English Integral Dataset, we did the same for the French Integral Dataset.
To do so, we kept as training the 10 first chunks of 500M of data.
The two following chunks of 500M of data were respectively used for validation (dev) and testing (test).

```shell
split --verbose -b500M data
```

The tokenizer was trained on the train set.

## English Training data: the Integral Dataset

The Integral Dataset (train, test and dev) is directly available at [English Integral Dataset](https://osf.io/jzcvu/).
The following steps can regenerate it (computationally expensive...)

### Cleaning data

```shell
# To apply to a single text file
python scripts/clean_gutenberg_books.py --path /path/to/your/text_file.txt 

# To apply on all Gutenberg books (you should load all the required books before)
python scripts/clean_gutenberg_books.py
```

### Merging books data

TODO

### Splitting data

```shell
# To apply it to a single text file
python scripts/scripts/split_dataset.py --path data/train.txt --nsplits 5
```

## fMRI data: The Little Prince

Load it from [TLP data](https://openneuro.org/datasets/ds003643/versions/2.0.1)
Should be saved in `data` as:

```shell
fMRI
 |- english
 |    |- sub-057
 |    |   |- func
 |    |   |   |- fMRI_english_sub-057_run1.nii.nii
 |    |   |   |- fMRI_english_sub-057_run2.nii.nii
 |    |   |   |- ...
 |    |   |   |- fMRI_english_sub-057_run9.nii.nii
 |    |- sub-058
 |    |   |- func
 |    |   |   |- fMRI_english_sub-058_run1.nii.nii
 |    |   |   |- fMRI_english_sub-058_run2.nii.nii
 |    |   |   |- ...
 |    |   |   |- fMRI_english_sub-058_run9.nii.nii
 |    |- ...
 |    |   |- ...
```

If you change the names or the structure, you should change the functions that retrieve the data in `src/irnlm/data`.

# Creating Semantic and Syntactic datasets

```shell
# Integral features Tokenization
python scripts/tokenize_dataset.py --path data/train_split-1.txt --type integral
# Repeat for all splits of train/test/dev

# Semantic features extraction
python scripts/create_semantic_dataset.py --text data/train_split-1.txt --saving_path data/train_semantic_split-1.txt
# Semantic Tokenization
python scripts/tokenize_dataset.py --path data/train_semantic_split-1.txt --type semantic
# Repeat for all splits of train/test/dev

# Syntactic features extraction
python scripts/create_syntactic_dataset.py --text data/train_split-1.txt --saving_path data/train_syntactic_split-1.pkl
# Syntactic Tokenization
python scripts/tokenize_dataset.py --path data/train_syntactic_split-1.pkl --type syntactic
# Repeat for all splits of train/test/dev

```

# Train Glove

```shell
# GloVe Integral
cd src/irnlm/models/glove/
./train.sh

# GloVe Semantic
cd src/irnlm/models/glove_semantic/
./train.sh

# GloVe Syntactic
cd src/irnlm/models/glove_syntactic/
./train.sh

```

# Train GPT-2

To train a GPT-2 model you should fill in a template in `src/irnlm/models/gpt2/templates/`, and call `src/irnlm/models/gpt2/train.py`.

```shell
python src/irnlm/models/gpt2/train.py --yaml_file src/irnlm/models/gpt2/templates/lm_template4_integral.yml
```


# Extract features from GloVe

```shell
python scripts/extract_features.py  --path data/train.txt \
                                    --model glove \
                                    --type integral \
                                    --context_size 507 \
                                    --max_seq_length 512 \
                                    --config src/irnlm/models/glove/GloVe_Integral.txt \
                                    --bsz 32 \
                                    --language english \
                                    --saving_path derivatives/features.csv 
python scripts/extract_features.py  --path data/train.txt \
                                    --model glove \
                                    --type semantic \
                                    --context_size 507 \
                                    --max_seq_length 512 \
                                    --config src/irnlm/models/glove_semantic/GloVe_Semantic.txt \
                                    --bsz 32 \
                                    --language english \
                                    --saving_path derivatives/features_semantic.csv 
python scripts/extract_features.py  --path data/train.txt \
                                    --model glove \
                                    --type syntactic \
                                    --context_size 507 \
                                    --max_seq_length 512 \
                                    --config src/irnlm/models/glove_syntactic/GloVe_Syntactic.txt \
                                    --bsz 32 \
                                    --language english \
                                    --saving_path derivatives/features_syntactic.csv 
```

# Extract features from GPT-2

TODO
```shell
python scripts/extract_features.py  --path data/train.txt \
                                    --model gpt2 \
                                    --type integral \
                                    --context_size 507 \
                                    --max_seq_length 512 \
                                    --config  TODO \
                                    --bsz 32 \
                                    --language english \
                                    --saving_path derivatives/features.csv \
```

# Fit brain data: the encoding model

A simple and easy to use package (fmri_encoder) can be loaded to fit linear encoding models:

```shell
git clone git@github.com:AlexandrePsq/fmri_encoder.git
cd fmri_encoder
pip install -r requirements.txt
pip install -e .
```

However, if you want to use the same code that was used in the paper, you should use `irnlm/encoding_pipeline`.
The code might be more abstract (as it implements several classes) and slower (as it performs a nested-cross validation).

```shell
# You should first fill the template `--yaml_file` depending on the features you want to use 
# and your choice of hyperparameters .
python src/irnlm/encoding_pipeline/main.py --yaml_file src/irnlm/encoding_pipeline/template.yml
```

# Running the analyses

All scripts are listed in :

```shell
python scripts/replicate_figX.py
```


# Citation

To cite this work, use:

```shell
@misc{pasquiou2023,
  doi = {10.48550/ARXIV.2302.14389},
  url = {https://arxiv.org/abs/2302.14389},
  author = {Pasquiou, Alexandre and Lakretz, Yair and Thirion, Bertrand and Pallier, Christophe},
  keywords = {Computation and Language (cs.CL), FOS: Computer and information sciences, FOS: Computer and information sciences},
  title = {Information-Restricted Neural Language Models Reveal Different Brain Regions' Sensitivity to Semantics, Syntax and Context},
  publisher = {arXiv},
  year = {2023},
  copyright = {Creative Commons Attribution 4.0 International}
}
```