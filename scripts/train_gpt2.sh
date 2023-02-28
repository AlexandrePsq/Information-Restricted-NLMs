# GPT-2 Integral
python src/irnlm/models/gpt2/train.py --yaml_file src/irnlm/models/gpt2/templates/lm_template4_full.yml

# GPT-2 Integral - Context 5-15-45
python src/irnlm/models/gpt2/train.py --yaml_file src/irnlm/models/gpt2/templates/lm_template4_context-5.yml
python src/irnlm/models/gpt2/train.py --yaml_file src/irnlm/models/gpt2/templates/lm_template4_context-15.yml
python src/irnlm/models/gpt2/train.py --yaml_file src/irnlm/models/gpt2/templates/lm_template4_context-45.yml

# GPT-2 Semantic
python src/irnlm/models/gpt2/train.py --yaml_file src/irnlm/models/gpt2_semantic/templates/lm_template4_semantic.yml

# GPT-2 Syntactic
python src/irnlm/models/gpt2/train.py --yaml_file src/irnlm/models/gpt2_syntactic/templates/lm_template4_syntax-context-full.yml
