from torchvision import transforms

from torch.utils.data import DataLoader
from scripts.inference.inference_dataset import InferenceDataset

def get_inference_dataloader(
    cfg: dict,
    img_dir = str,
    batch_size: int = 8,
) -> DataLoader:
    """
    Creates a Dataloader used for inference

    Args:
        cfg: Dictionary of parsed cfg.yaml file.
        img_dir: Directory of images for inference
        batch_size (int): Number of samples per batch. Default is 8.

    Returns:
        Inference DataLoader: A PyTorch DataLoader that samples from the inference dataset.
    """
    
    transform = transforms.Compose([
    transforms.Resize(cfg['transforms']['resize']),
    transforms.ToTensor(),
    transforms.Normalize(cfg['transforms']['normalize']['mean'], cfg['transforms']['normalize']['std'])
    ])

    ds = InferenceDataset(
        img_dir,
        transforms=transform,
    )

    dataloader = DataLoader(
        ds,
        batch_size=batch_size,
    )

    return dataloader
