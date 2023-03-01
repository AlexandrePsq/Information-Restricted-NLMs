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

## Training data: the Integral Dataset

The Integral Dataset (train, test and dev) is directly available at https://osf.io/jzcvu/
The following steps can regenerate it (computatinoal expensive...)

### Cleaning data

```shell
# To apply it to a single text file
python scripts/clean_gutenberg_books.py --path /path/to/your/text_file.txt 

# To apply it on all Gutenberg books (you should load all the required books before)
python scripts/clean_gutenberg_books.py
```

### Merging books data

TODO

## Splitting data

```shell
# To apply it to a single text file
python scripts/scripts/split_dataset.py --path data/train.txt --nsplits 5
```

## fMRI data: The Little Prince

Load it from https://openneuro.org/datasets/ds003643/versions/2.0.1
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

```shell
# GPT-2 Integral
python src/irnlm/models/gpt2/train.py --yaml_file src/irnlm/models/gpt2/templates/lm_template4_full.yml

# GPT-2 Integral - Context 5-15-45
python src/irnlm/models/gpt2/train.py --yaml_file src/irnlm/models/gpt2/templates/lm_template4_context-5.yml
python src/irnlm/models/gpt2/train.py --yaml_file src/irnlm/models/gpt2/templates/lm_template4_context-15.yml
python src/irnlm/models/gpt2/train.py --yaml_file src/irnlm/models/gpt2/templates/lm_template4_context-45.yml

# GPT-2 Semantic
python src/irnlm/models/gpt2/train.py --yaml_file src/irnlm/models/gpt2_semantic/templates/lm_template4_full.yml

# GPT-2 Syntactic
python src/irnlm/models/gpt2/train.py --yaml_file src/irnlm/models/gpt2_syntactic/templates/lm_template4_syntax-context-full.yml

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
# You shoudl first fill the template `yaml_file`depending on the features you want to use 
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

```python
@article{PasquiouIRNLM2023,
  author = {... to do},
  doi = {},
  month = {02},
  title = {{"Information-Restricted Neural Language Models Reveal Different Brain Regions' Sensitivity to Semantics, Syntax and Context"}},
  url = {https://github.com/AlexandrePsq/Information-Restrited-NLMs},
  year = {2023}
}
```