import os
import yaml

def read_config() -> dict:
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    cfg_path = os.path.join(project_root, 'cfg.yaml')
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg