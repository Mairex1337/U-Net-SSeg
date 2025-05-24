import torch

from src.utils import read_config, get_device, get_model


def load_model() -> torch.nn.Module:
    """
    Loads the model which will be used for inference.
    
    Returns:
        torch.nn.Module: initialized model
    """    
    cfg = read_config()
    device = get_device()
    model = get_model(cfg, 'baseline').to(device)
    
    model_path = cfg['inference']['model_path']
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint["model_state_dict"])
    
    return model