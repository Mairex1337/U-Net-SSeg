import os
import re
from typing import Literal

import yaml


def read_config() -> dict:
    """
    Load the project's configuration file as a dictionary.

    Returns:
        dict: Parsed configuration from cfg.yaml.
    """
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    cfg_path = os.path.join(project_root, 'cfg.yaml')
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg

def write_config(cfg: dict) -> None:
    """
    Write the configuration dictionary to cfg.yaml.

    Args:
        cfg (dict): Dictionary of the cfg file

    Returns:
        None
    """
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    cfg_path = os.path.join(project_root, 'cfg.yaml')
    with open(cfg_path, "w") as f:
        yaml.safe_dump(cfg, f, default_flow_style=False)


def resolve_path(path: str) -> str:
    """
    Resolves a relative path to an absolute path from the project root.

    Args:
        path (str): Relative path using forward slashes ('/') from project root.

    Returns:
        str: Absolute resolved path.

    Raises:
        FileNotFoundError: If the resolved path does not exist.
    """
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    abs_path = os.path.join(project_root, *path.split('/'))
    if not os.path.exists(abs_path):
        raise FileNotFoundError(
            f"Resolved path does not exist: {abs_path}.",
            "Check cfg.yaml for appropriate /data dir structure."
        )
    return abs_path


def get_run_dir(run_id: str, model_name: str) -> str:
    """
    Returns the full path to a run directory for a given model and run ID.

    Args:
        run_id (str): Unique ID of the run.
        model_name (str): Name of the model.

    Returns:
        str: Full path to the run directory.
    """
    if "SLURM_JOBID" in os.environ:
        base_path = os.environ["TMPDIR"]
    else:
        base_path = resolve_path("")
    run_directory = os.path.join(base_path, "outputs", model_name, run_id)
    os.makedirs(run_directory, exist_ok=True)
    return run_directory


def get_best_checkpoint(checkpoints_dir: str) -> str:
    """
    Finds and returns the path to the best checkpoint in a given directory.

    Args:
        checkpoints_dir (str): Path to the directory containing model checkpoints.

    Returns:
        str: Full path to the checkpoint file that contains 'best' in its name.

    Raises:
        FileNotFoundError: If the checkpoint directory doesn't exist
                           or no 'best' checkpoint is found.
    """
    if not os.path.exists(checkpoints_dir):
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoints_dir}")

    best_checkpoint = None
    for file in os.listdir(checkpoints_dir):
        if "best" in file and file.endswith(".pth"):
            best_checkpoint = os.path.join(checkpoints_dir, file)
            break

    if best_checkpoint is None:
        raise FileNotFoundError(f"No checkpoint with 'best' in the name found in {checkpoints_dir}")

    return best_checkpoint

def get_best_loss(
        run_dir: str,
        initial_run_id: int
) -> Literal['weighted_cle', 'OHEMLoss', 'mixed_cle_dice', 'dice']:
    loss_map = {
        0: 'weighted_cle',
        1: 'OHEMLoss',
        2: 'mixed_cle_dice',
        3: 'dice'
    }
    best_eval = -float('inf')
    best_loss = ''
    for idx, run_id in enumerate(range(initial_run_id, initial_run_id + 4)):
        log_path = os.path.join(run_dir, str(run_id), 'eval.log')
        result_str = open(log_path, 'r').read()
        pattern = r"Mean Accuracy:\s*(\d+\.\d*)\n.*Mean IoU:\s*(\d+\.\d*)\n.*Mean Dice \(F1\):\s*(\d+\.\d*)"
        match = re.search(pattern, result_str)
        metric = float(match.group(1)) + float(match.group(2)) + float(match.group(3))
        if metric > best_eval:
            best_eval = metric
            best_loss = loss_map[idx]
    
    assert best_loss != ''
    print(best_loss)
