import torch
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import sys
sys.path.insert(1, '')
from copy import deepcopy
from src.data.dataset import SegmentationDataset
import yaml

def calculate_mean_std() -> None:
    """
    
    """
    transform = transforms.Compose([
        transforms.Resize((456,256)),
        transforms.ToTensor()
    ])

    train_dataset = SegmentationDataset("data\\bdd100k\\images\\10k\\train\\", "data\\bdd100k\\labels\\sem_seg\\colormaps\\train\\", img_transforms=transform, mask_transforms=transform)
    train_loader = DataLoader(train_dataset)

    channel_sums = torch.zeros(3)
    channel_sqr_sums = torch.zeros(3)
    total_pixels = 0

    for image, _ in train_loader:
        image = image.squeeze()
        _,h,w = image.shape
        total_pixels += h*w
        channel_sums += image.sum(dim=[1,2])
        channel_sqr_sums += (image**2).sum(dim=[1,2])
        
    mean = channel_sums / total_pixels
    var = (channel_sqr_sums / total_pixels) - (mean**2)
    std = torch.sqrt(var)
    
    mean = mean.tolist()
    
    std = std.tolist()

    with open('src\\cfg.yaml', 'r') as f:
        cfg = yaml.unsafe_load(f)

    cfg['normalization'] = {'mean': deepcopy(mean), 'std': deepcopy(std)}
    
    with open('src\\cfg.yaml', 'w') as f:
        cfg = yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
        
calculate_mean_std()
