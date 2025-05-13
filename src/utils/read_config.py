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