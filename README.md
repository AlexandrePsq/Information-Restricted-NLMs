# Information-Restrited-NLMs
Code of the paper: "Information-Restricted Neural Language Models Reveal Different Brain Regions' Sensitivity to Semantics, Syntax and Context"

# Installation & Set up

First, install the requirements.

```shell
git clone git@github.com:AlexandrePsq/Information-Restrited-NLMs.git
cd Information-Restrited-NLMs
pip install -r requirements.txt
pip install -e .
```

# Loading data

## Training data: the Integral Dataset

TODO

## fMRI data: The Little Prince

TODO

# Cleaning data

TODO

# Splitting data

TODO

# Creating Semantic and Syntactic datasets

TODO

# Train Glove

TODO

# Train GPT-2

TODO

# Extract features from GloVe

TODO

# Extract features from GPT-2

TODO


# Fit brain data: the encoding model

Load fmri_encoder:

```shell
git clone git@github.com:AlexandrePsq/fmri_encoder.git
cd fmri_encoder
pip install -r requirements.txt
pip install -e .
```

TODO

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