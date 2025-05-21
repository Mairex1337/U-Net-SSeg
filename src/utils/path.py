import os

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
    """Get the run directory for saving logs and checkpoints"""
    base_path = resolve_path("outputs/")
    run_directory = os.path.join(base_path, model_name, run_id)
    os.makedirs(run_directory, exist_ok=True)
    return run_directory


def get_best_checkpoint(checkpoints_dir: str) -> str:
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
