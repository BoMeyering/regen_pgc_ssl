# Dataloader functionality
# BoMeyering 2024

import math
import torch
import random
import numpy as np
from torch.utils.data import DataLoader, Sampler
from typing import Iterable

class InfiniteSampler(Sampler):
    def __init__(self, dataset: torch.utils.data.Dataset):
        self.indices = list(range(len(dataset)))
        self.dataset = dataset

    def __iter__(self):
        random.shuffle(self.indices)
        while True:
            for i in self.indices:
                yield i % len(self.indices)

    def __len__(self):
        return len(self.dataset)

class DataLoaderBalancer:
    def __init__(self, *datasets, batch_sizes: Iterable[int, ], drop_last: bool=True):
        self.datasets = datasets
        self.batch_sizes = batch_sizes
        self.drop_last = drop_last
        
        self.dl_lengths = []
        self.dataloaders = []

        if len(self.datasets) != len(self.batch_sizes):
            raise ValueError("The number of datasets does not equal the number of batch sizes. Please ammend appropriately")
        for i, (ds, bs) in enumerate(zip(self.datasets, self.batch_sizes)):
            if len(ds) < bs:
                raise ValueError(f"Dataset {i+1} has fewer elements than its specified batch size. Please select a batch size smaller than {bs} and try again.")

        if drop_last:
            for i, ds in enumerate(self.datasets):
                self.dl_lengths.append(len(ds) // self.batch_sizes[i])
        else:
            for i, ds in enumerate(self.datasets):
                self.dl_lengths.append(math.ceil(len(ds) / self.batch_sizes[i]))
                
    def balance_loaders(self):
        # Get the index of the longest dataloader
        self.maxdl = np.argmax(self.dl_lengths)
        for i, ds in enumerate(self.datasets):
            # For the longest dataloader, create a loader that iterates over everything once
            if i == self.maxdl:
                self.dataloaders.append(DataLoader(ds, batch_size=self.batch_sizes[i], shuffle=True, drop_last=self.drop_last))
            else: # Wrap the rest of the dataloaders with InfiniteSampler
                self.dataloaders.append(DataLoader(ds, batch_size=self.batch_sizes[i], sampler=InfiniteSampler(ds), drop_last=self.drop_last))
        
        return self.dataloaders, self.dl_lengths[self.maxdl]