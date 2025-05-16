import logging
import os
from src.utils.resolve_path import resolve_path


def get_run_dir(run_id: str, model_name: str) -> str:
    """
    Get the run directory for saving logs and checkpoints.
    """
    base_path = resolve_path("outputs/", 2)
    run_directory = os.path.join(base_path, model_name, run_id)
    os.makedirs(run_directory, exist_ok=True)
    return run_directory

def setup_logging(run_dir: str) -> logging.Logger:
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