""" 
"""
import warnings

warnings.simplefilter(action="ignore")

import os
import torch
import pickle

from torch.utils.data import TensorDataset
from torch.utils.data import (
    DataLoader,
    RandomSampler,
    SequentialSampler,
    DistributedSampler,
)

from irnlm.models.gpt2.dataset import Dataset
from irnlm.models.gpt2.processors import DataProcessor
from irnlm.utils import check_folder


class LMDataset(Dataset):
    """Class for language modeling dataset fetching and formatting."""

    def __init__(self, task_name, dataset_dir=None, url=None, language="english"):
        super(LMDataset, self).__init__(task_name, dataset_dir, url)
        self.language = language

    def get_labels(self):
        """Returns possible labels for the task."""
        raise NotImplementedError()


class LMProcessor(DataProcessor):
    """Processor for language modeling."""

    def __init__(
        self,
        train_paths,
        dev_paths,
        test_paths,
        max_seq_length,
        device="cpu",
        output_dir="./",
        dataset_dir="./",
        n_splits=5,
        context_size=None,
    ):
        self.max_seq_length = max_seq_length
        print(
            f"Using context_size of: {context_size} and max_seq_length of {self.max_seq_length}"
        )
        self.device = device
        self.output_dir = output_dir
        self.dataset_dir = dataset_dir
        self.n_splits = n_splits
        self.context_size = context_size
        self.train_paths = train_paths
        self.dev_paths = dev_paths
        self.test_paths = test_paths

    def get_data(self, set_type):
        """See base class."""
        # paths = [os.path.join(self.dataset_dir, f'{self.dataset_name}{self.extra}{set_type}_all-ids_split-{index_split}.pkl') for index_split in range(self.n_splits)]
        if set_type == "train":
            paths = self.train_paths
        elif set_type == "dev":
            paths = self.dev_paths
        elif set_type == "test":
            paths = self.test_paths
        else:
            raise ValueError(f"set_type {set_type} unknown.")
        if all([os.path.exists(p) for p in paths]):
            return paths
        else:
            raise NotImplementedError("Paths not defined: ", paths)

    def set_tokenizer(self, tokenizer):
        """Set processor tokenizer."""
        self.tokenizer = tokenizer

    def save_object(self, filename, data):
        """Save computed examples and features."""
        with open(filename, "wb") as outp:  # Overwrites any existing file.
            pickle.dump(data, outp, pickle.HIGHEST_PROTOCOL)

    def load_object(self, filename):
        """Load computed examples and features."""
        with open(filename, "rb") as inp:  # Overwrites any existing file.
            data = pickle.load(inp)
        return data

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
        # attention_mask = None #torch.cat([f.attention_mask for f in features], dim=0)
        # token_type_ids =  torch.cat([f.token_type_ids for f in features], dim=0)
        # label_ids =  torch.cat([f.label_ids for f in features], dim=0)
        data = TensorDataset(
            input_ids
        )  # attention_mask, token_type_ids, label_ids were removed !
        if set_type == "train":
            if local_rank == -1:
                sampler = RandomSampler(data)
            else:
                sampler = DistributedSampler(data)
        else:
            sampler = SequentialSampler(data)
        dataloader = DataLoader(data, sampler=sampler, batch_size=batch_size)
        # shuffle = (set_type=='train')
        # dataloader = DataLoader(data, shuffle=shuffle, batch_size=batch_size)
        return dataloader
