# Dataloader functionality
# BoMeyering 2024

import math
import torch
import random
import numpy as np
from torch.utils.data import DataLoader, Sampler, Dataset
from torch.utils.data.distributed import DistributedSampler
from typing import Iterable, Optional
import torch.distributed as dist

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

        self.maxdl = np.argmax(self.dl_lengths)
                
    def balance_loaders(self):
        # Get the index of the longest dataloader
        for i, ds in enumerate(self.datasets):
            # For the longest dataloader, create a loader that iterates over everything once
            if i == self.maxdl:
                self.dataloaders.append(DataLoader(ds, batch_size=self.batch_sizes[i], shuffle=True, drop_last=self.drop_last))
            else: # Wrap the rest of the dataloaders with InfiniteSampler
                self.dataloaders.append(DataLoader(ds, batch_size=self.batch_sizes[i], sampler=InfiniteSampler(ds), drop_last=self.drop_last))
        
        return self.dataloaders, self.dl_lengths[self.maxdl]
    
class DistributedInfiniteSampler(Sampler):
    def __init__(self, dataset: Dataset, num_replicas: Optional[int] = None, rank: Optional[int]=None, shuffle: bool=True, seed: int=0):
        if num_replicas is not None:
            if not dist.is_available():
                raise RuntimeError("Requires distribbuted package to be available")
            num_replicas = dist.get_world_size()
        if rank is not None:
            if not dist.is_available():
                raise RuntimeError("Requiires distributed package to be available")
            rank = dist.get_rank()
        if rank >= num_replicas or rank < 0:
            raise ValueError(f"Invalid rank {rank}, rank should be in the interval [0, {num_replicas - 1}]")
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.num_samples = int(math.ceil(len(self.dataset) / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas
        self.indices = list(range(len(self.dataset)))
        self.shuffle = shuffle
        self.seed = seed

    def __iter__(self):
        # Shuffle the indices before handling
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()

        # Remove tail of the indices to ensure it is evenly divisible
        indices = indices[:self.total_size]
        assert len(indices) == self.total_size

        indices = self.indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        while True:
            yield from indices

    def __len__(self):
        return self.num_samples
    
    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch
    
class DistributedDataloaderBalancer:
    def __init__(self, *datasets, batch_sizes: Iterable[int, ], num_replicas=None, rank=None):
        self.datasets = datasets
        self.batch_sizes = batch_sizes
        self.num_replicas = num_replicas if num_replicas is not None else dist.get_world_size()
        self.rank = rank if rank is not None else dist.get_rank()
        self.dl_lengths = []
        self.dataloaders = []

        if len(self.datasets) != len(self.batch_sizes):
            raise ValueError("The number of datasets does not equal the number of batch sizes. Please ammend appropriately")
        for i, (ds, bs) in enumerate(zip(self.datasets, self.batch_sizes)):
            if len(ds) < bs:
                raise ValueError(f"Dataset {i+1} has fewer elements than its specified batch size. Please select a batch size smaller than {bs} and try again.")
            self.dl_lengths.append(len(ds) // bs)

        self.maxdl = np.argmax(self.dl_lengths)

    def balance_loaders(self):
        for i, (ds, bs) in enumerate(zip(self.datasets, self.batch_sizes)):
            # For the longest dataloader, create a loader that iterates over everything once
            if i == self.maxdl:
                # Create a distributed sampler with DistributedSampler
                self.sampler = DistributedSampler(
                    dataset=ds,
                    num_replicas=self.num_replicas,
                    rank=self.rank,
                    shuffle=True,
                    drop_last=True
                )
                # Instantiate the dataloader
                dataloader = DataLoader(
                    dataset=ds,
                    batch_size=bs,
                    shuffle=False, 
                    sampler=self.sampler,
                    drop_last=True
                )
                # Append the dataloader
                self.dataloaders.append(dataloader)

            else: # Wrap the second dataloader with the DistributedInfiniteSampler
                self.infinite_sampler = DistributedInfiniteSampler(
                    dataset=ds,
                    num_replicas=self.num_replicas,
                    rank=self.rank,
                    shuffle=True
                )
                # Instantiate the dataloader
                dataloader = DataLoader(
                    dataset=ds,
                    batch_size=bs,
                    shuffle=False,
                    sampler=self.infinite_sampler,
                    drop_last=True
                )
                # Append the dataloader
                self.dataloaders.append(dataloader)
         
        return self.dataloaders, self.dl_lengths[self.maxdl]
        