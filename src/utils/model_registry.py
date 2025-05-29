from src.models import BaselineModel, UNet
import torch
from typing import Dict, Any
from src.utils.path import read_config, get_best_checkpoint, get_run_dir
from src.utils.env import get_device
import os

MODELS = {'baseline': BaselineModel, 'unet': UNet}  # Add other models as needed


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
    if model_name == 'unet':
        model = MODELS[model_name](
            in_channels=hyperparams['in_channels'],
            num_classes=hyperparams['num_classes'],
            base_channels=hyperparams['base_channels']
        )
    elif model_name == 'baseline':
        model = MODELS[model_name](
            hyperparams['input_dim'],
            hyperparams['hidden_dim'],
            hyperparams['num_classes']
        )
    return model


def load_model(run_id: str, model_name: str) -> torch.nn.Module:
    """
    Loads a trained model checkpoint for inference or evaluation.

    Args:
        run_id (str): The ID of the training run (used to locate the model checkpoint directory).
        model_name (str): The model identifier (used to build the correct model architecture).

    Returns:
        torch.nn.Module: The loaded model with weights from the checkpoint.
    """
    cfg = read_config()
    device = get_device()
    run_dir = get_run_dir(run_id,model_name)
    checkpoint_dir = os.path.join(run_dir, 'checkpoints')
    checkpoint_path = get_best_checkpoint(checkpoint_dir)
    model = get_model(cfg, model_name).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    return model
