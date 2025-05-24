from src.models import BaselineModel
import torch
from typing import Dict, Any

MODELS = {'baseline': BaselineModel}


def get_model(cfg: Dict[str, Any], model_name: str) -> torch.nn.Module:
    """
    Initializes the model based on configuration and model name.

    Args:
        cfg (Dict[str, Any]): Configuration dictionary loaded from YAML.
        model_name (str): Key for selecting the model type.

    Returns:
        torch.nn.Module: Initialized model (not yet moved to device).
    """
    assert model_name in MODELS, f"Model '{model_name}' not supported."
    hyperparams = cfg['hyperparams'][f'{model_name}']
    model = MODELS[model_name](
        hyperparams['input_dim'],
        hyperparams['hidden_dim'],
        hyperparams['num_classes']
    )
    return model
