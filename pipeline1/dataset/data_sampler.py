# In this file everything related to dataloading and sampling is gathered
import torch
from torch.utils.data import Dataset, Sampler
from torch import distributed as dist
import math
import numpy as np
import random


class DatasetFromSampler(Dataset):
    """Dataset to create indexes from `Sampler`.
    Args:
        sampler: PyTorch sampler
    """

    def __init__(self, sampler: Sampler):
        """Initialisation for DatasetFromSampler."""
        self.sampler = sampler
        self.sampler_list = None
        if self.sampler_list is None:
            self.sampler_list = list(self.sampler)

    def __getitem__(self, index: int):

        ret = self.sampler_list[index]
        return ret

    def __len__(self) -> int:
        return len(self.sampler)


class RandomSampler(Sampler):
    def __init__(self, len_data, i=0, PARAMS={}):
        self.seq = np.arange(len_data, dtype=np.int32)
        random.seed(PARAMS["seed"])
        np.random.seed(PARAMS["seed"])
        np.random.shuffle(self.seq)
        self.seq = self.seq[i * PARAMS["batch_size"] :]

        if PARAMS["distributed"]:
            self.num_replicas = dist.get_world_size()
            self.rank = dist.get_rank()

            self.num_samples = math.ceil(len(self.seq) / self.num_replicas)
            self.total_size = self.num_samples * self.num_replicas

            self.seq = self.seq[self.rank : self.total_size : self.num_replicas]

        # print(self.seq)

    def __iter__(self):
        return iter(self.seq)

    def __len__(self):
        return len(self.seq)


class OrderedDistributedSampler(Sampler):
    """Sampler that restricts data loading to a subset of the dataset.
    It is especially useful in conjunction with
    :class:`torch.nn.parallel.DistributedDataParallel`. In such case, each
    process can pass a DistributedSampler instance as a DataLoader sampler,
    and load a subset of the original dataset that is exclusive to it.
    .. note::
        Dataset is assumed to be of constant size.
    Arguments:
        dataset: Dataset used for sampling.
        num_replicas (optional): Number of processes participating in
            distributed training.
        rank (optional): Rank of the current process within num_replicas.
    """

    def __init__(self, dataset, num_replicas=None, rank=None):
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        indices = list(range(len(self.dataset)))

        # add extra samples to make it evenly divisible
        indices += indices[: (self.total_size - len(indices))]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank : self.total_size : self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self):
        return self.num_samples


def reduce_tensor(tensor, n):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= n
    return rt


def sync_across_gpus(t, world_size):
    torch.distributed.barrier()
    gather_t_tensor = [torch.ones_like(t) for _ in range(world_size)]
    torch.distributed.all_gather(gather_t_tensor, t)
    return torch.cat(gather_t_tensor)