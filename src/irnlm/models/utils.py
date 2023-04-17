import os
import re
import glob
import torch
import random
import datetime
import numpy as np
from transformers import GPT2LMHeadModel, WEIGHTS_NAME, CONFIG_NAME


from irnlm.models.gpt2.modeling_hacked_gpt2_integral import GPT2LMHeadModel
from irnlm.models.gpt2.modeling_hacked_gpt2_semantic import GPT2LMHeadModel as GPT2LMHeadModelSemantics
from irnlm.models.gpt2.modeling_hacked_gpt2_syntactic import GPT2LMHeadModel as GPT2LMHeadModelSyntax



def set_seed(value=1111):
    """ Set all seeds to a given value for reproductibility."""
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(value)

def format_time(elapsed):
    """
    Takes a time in seconds and returns a string hh:mm:ss
    """
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))
    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))

def get_device(device_number=0, local_rank=-1):
    """ Get the device to use for computations.
    """
    if local_rank == -1:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        n_gpu = torch.cuda.device_count()
        print('There are %d GPU(s) available.' % torch.cuda.device_count())
        if torch.cuda.is_available():
            print('We will use the GPU:', torch.cuda.get_device_name(device_number))
        else:
            print('No GPU available, using the CPU instead.')
    else:
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')
    return device

def tryint(s):
    try:
        return int(s)
    except:
        return s

def alphanum_key(s):
    """ Turn a string into a list of string and number chunks.
        "z23a" -> ["z", 23, "a"]
    """
    return [ tryint(c) for c in re.split('([0-9]+)', s) ]

def sort_nicely(l):
    """ Sort the given list in the way that humans expect.
    """
    l.sort(key=alphanum_key)

def load_last_checkpoint(parameters, model=None, gpt2_type='integral'):
    """Load the last saved model in case it has crashed...
    Args:
        - parameters: dict
    Returns:
        - model: GPT2LMHeadModel
        - start_at_dataloader: int
    """
    start_at_dataloader = 0
    path = glob.glob(os.path.join(parameters['output_dir'], 'end-epoch-*'))
    sort_nicely(path)
    path_loader = glob.glob(os.path.join(parameters['output_dir'], 'end-epoch-*_split*'))
    sort_nicely(path_loader)
    if (len(path)==0) and (len(path_loader)==0):
        path = glob.glob(os.path.join(parameters['output_dir'], 'start-epoch-*'))
        sort_nicely(path)
    elif os.path.basename(path[-1]).split('epoch-')[-1]==os.path.basename(path_loader[-1]).split('epoch-')[-1].split('_split')[0]:
        path = path[-1]
    else:
        path = path_loader[-1]
        start_at_dataloader = os.path.basename(path_loader[-1]).split('epoch-')[-1].split('_split-')[-1]            

    try:
        if gpt2_type=='integral':
            model = GPT2LMHeadModel.from_pretrained(
                            path,
                            output_attentions=parameters['output_attentions'], # Whether the model returns attentions weights.
                            output_hidden_states=parameters['output_hidden_states'], # Whether the model returns all hidden-states.
            )
        elif gpt2_type=='semantic':
            model = GPT2LMHeadModelSemantics.from_pretrained(
                            path,
                            output_attentions=parameters['output_attentions'], # Whether the model returns attentions weights.
                            output_hidden_states=parameters['output_hidden_states'], # Whether the model returns all hidden-states.
            )
        elif gpt2_type=='syntactic':
            model = GPT2LMHeadModelSyntax.from_pretrained(
                            path,
                            output_attentions=parameters['output_attentions'], # Whether the model returns attentions weights.
                            output_hidden_states=parameters['output_hidden_states'], # Whether the model returns all hidden-states.
            )
        else:
            raise NotImplementedError('Model name not recognized...')
        print(f'Using model saved at: {path}...')
    except:
        print(f'Using model created from scratch.')
    return model, int(start_at_dataloader)

def save_checkpoint(model, tokenizer, output_dir, index):
    """ Saving best-practices: if you use defaults names for the model, 
    you can reload it using from_pretrained().
    """
    output_dir = os.path.join(output_dir, index)
    # If we save using the predefined names, we can load using `from_pretrained`
    output_model_file = os.path.join(output_dir, WEIGHTS_NAME)
    output_config_file = os.path.join(output_dir, CONFIG_NAME)
    # Create output directory if needed
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    print("Saving model to %s" % output_dir)

    # Save a trained model, configuration and tokenizer using `save_pretrained()`.
    # They can then be reloaded using `from_pretrained()`
    model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
    torch.save(model_to_save.state_dict(), output_model_file)
    model_to_save.config.to_json_file(output_config_file)
    #tokenizer.save_pretrained(output_dir)
