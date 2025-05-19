import torch.cuda as cuda
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