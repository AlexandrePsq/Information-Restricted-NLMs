""" 
"""
import warnings
warnings.simplefilter(action='ignore')

import os
import torch
import pickle

from torch.utils.data import TensorDataset
from torch.utils.data import (
    DataLoader, 
    RandomSampler, 
    SequentialSampler, 
    DistributedSampler
)

from irnlm.models.gpt2.dataset import Dataset
from irnlm.models.gpt2.processors import DataProcessor
from irnlm.utils import check_folder


class LMDataset(Dataset):
    """Class for language modeling dataset fetching and formatting."""

    def __init__(self, task_name, dataset_name, dataset_dir=None, url=None, language='english', extra=''):
        super(LMDataset, self).__init__(task_name, dataset_name, dataset_dir, url, extra=extra)
        self.language = language

    def _fetch_dataset(self):
        """Fetch sentence classification dataset."""
        if not os.path.exists(self.dataset_dir):
            check_folder(self.dataset_dir)
            if self.dataset_name=='lpp':
                pass
    
    def get_labels(self):
        """ Returns possible labels for the task.
        """
        raise NotImplementedError()


class LMProcessor(DataProcessor):
    """Processor for language modeling."""
    
    def __init__(self, max_seq_length, device='cpu', output_dir='./', dataset_name='', dataset_dir='./', n_splits=5, context_size=None, extra=''):
        self.max_seq_length = max_seq_length if context_size is None else context_size+5 # +5 because of the special tokens + the current and following tokens
        print(f'Using context_size of: {context_size} and max_seq_length of {self.max_seq_length}')
        self.device = device
        self.dataset_name = dataset_name
        self.output_dir = output_dir
        self.dataset_dir = dataset_dir
        self.n_splits = n_splits
        self.context_size=context_size
        self.extra = extra

    def get_data(self, set_type):
        """See base class."""
        if all([os.path.exists(os.path.join(self.dataset_dir, f'{self.dataset_name}{self.extra}{set_type}_all-ids_split-{index_split}.pkl')) for index_split in range(self.n_splits)]):
            paths = [os.path.join(self.dataset_dir, f'{self.dataset_name}{self.extra}{set_type}_all-ids_split-{index_split}.pkl') for index_split in range(self.n_splits)]

        else:
            tmp = [os.path.exists(os.path.join(self.dataset_dir, f'{self.dataset_name}{self.extra}{set_type}_all-ids_split-{index_split}.pkl')) for index_split in range(self.n_splits)]
            raise NotImplementedError("Dataset not defined: ", tmp)
        return paths
    
    def set_tokenizer(self, tokenizer):
        """Set processor tokenizer."""
        self.tokenizer = tokenizer
        
    def save_object(self, filename, data):
        """Save computed examples and features.
        """
        with open(filename, 'wb') as outp:  # Overwrites any existing file.
            pickle.dump(data, outp, pickle.HIGHEST_PROTOCOL)
    
    def load_object(self, filename):
        """Load computed examples and features.
        """
        with open(filename, 'rb') as inp:  # Overwrites any existing file.
            data = pickle.load(inp)
        return data
    
    def create_examples(self, sequence):
        """Returns list of InputExample objects."""
        input_id = self.pad_to_max_length([0] + sequence + [1, 2]) ### HERE ###
        #attention_mask = self.pad_attention_to_max_length([1] + sequence + [1, 1])
        return input_id #, attention_mask
            
    def pad_to_max_length(self, sequence):
        """Pad sequence to reach max_seq_length"""
        n = len(sequence)
        if n==self.max_seq_length:
            return sequence
        else:
            print(f'Careful - {sequence} - is not of {len(sequence)} (!= max length)... Padding...')
            sequence = sequence[:self.max_seq_length]
            result = sequence + [1] * ((self.max_seq_length - n)// 2) ### HERE ###
            if len(result)==self.max_seq_length:
                return result
            else:
                return result + [1] ### HERE ###
        
    def pad_attention_to_max_length(self, sequence):
        """Pad sequence to reach max_seq_length"""
        sequence = sequence[:self.max_seq_length]
        n = len(sequence)
        result = [1 for _ in sequence] + [0, 0] * ((self.max_seq_length - n)// 2)
        if len(result)==self.max_seq_length:
            return result
        else:
            return result + [0]
    
    def get_data_loader(self, features_path, batch_size, local_rank, set_type):
        """See base class."""
        # Loading features split
        if isinstance(features_path, str):
            features = self.load_object(features_path)
        else:
            features = [self.load_object(path) for path in features_path]
            features = [item for l in features for item in l]
        
        # Creating data loader
        input_ids = torch.cat([f.input_ids for f in features], dim=0)
        #attention_mask = None #torch.cat([f.attention_mask for f in features], dim=0)
        #token_type_ids =  torch.cat([f.token_type_ids for f in features], dim=0)
        #label_ids =  torch.cat([f.label_ids for f in features], dim=0)
        data = TensorDataset(input_ids) # attention_mask, token_type_ids, label_ids were removed !
        if set_type=='train':
            if local_rank == -1:
                sampler = RandomSampler(data)
            else:
                sampler = DistributedSampler(data)
        else:
            sampler = SequentialSampler(data)
        dataloader = DataLoader(data, sampler=sampler, batch_size=batch_size)
        #shuffle = (set_type=='train')
        #dataloader = DataLoader(data, shuffle=shuffle, batch_size=batch_size)
        return dataloader
