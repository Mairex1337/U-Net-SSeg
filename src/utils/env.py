import os

import torch.cuda as cuda
import torch.distributed as dist
import torch.mps as mps


def get_device() -> 'str':
    """Returns the best available device: 'cuda', 'mps', or 'cpu'."""
    if cuda.is_available():
        device = 'cuda'
    elif mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'
    return device

def setup_ddp_process(rank: int, world_size: int) -> None:
    """
    Set up ddp process for each GPU.

    Args:
        rank (int): Rank of the gpu.
        world_size (int): Number of total gpu's.

    Returns:
        None 
    """
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)
    cuda.set_device(rank)