import logging
import os

from torch.cuda import is_available as cuda_is_available
from torch.mps import is_available as mps_is_available

from src.utils.resolve_path import resolve_path


def get_run_dir(run_id: str, model_name: str) -> str:
    """
    Get the run directory for saving logs and checkpoints.
    """
    base_path = resolve_path("outputs/", 2)
    run_directory = os.path.join(base_path, model_name, run_id)
    os.makedirs(run_directory, exist_ok=True)
    return run_directory

def get_logger(run_dir: str) -> logging.Logger:
    log_file = os.path.join(run_dir, "training.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger("Trainer")
    return logger

def get_device() -> 'str':
    if cuda_is_available():
        device = 'cuda'
    elif mps_is_available():
        device = 'mps'
    else:
        device = 'cpu'
    return device