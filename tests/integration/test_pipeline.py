import numpy as np
import pytest
import torch
import yaml
from PIL import Image
from src.data.dataloader import get_dataloader
from src.utils.calculate_means import calculate_mean_std
from tests.integration.utils.toy_dataset import toy_dataset
from torchvision.transforms import functional as F
from torchvision.transforms.functional import to_pil_image


def _check_color_jitter(cfg):
    """
    """
    cfg["transforms"]["flip"] = 0.0  # Disable flip to isolate jitter
    dataloader = get_dataloader(cfg=cfg, train=True, batch_size=1, debug=False, stats=False)

    img1, _ = next(iter(dataloader))
    img2, _ = next(iter(dataloader))

    diff = torch.abs(img1 - img2).mean().item()
    assert diff > 0.01, "Color jitter didn't apply — images are too similar."


def _check_horizontal_flip(cfg: Dict[str, Any]) -> None:
    """
    Validates that horizontal flipping was applied correctly to the mask.

    This test forces horizontal flipping by setting the flip probability to 1.0.
    Instead of verifying the transformation on the image, it checks the mask,
    which is unaffected by color jitter or normalization. This ensures that the
    flip transformation is being applied correctly and can be evaluated using
    exact equality.

    Args:
        cfg (Dict[str, Any]): The dataset configuration dictionary containing transform parameters.

    Raises:
        AssertionError: If the flipped mask does not match the expected flipped version.
    """
    cfg["transforms"]["flip"] = 1.0

    dataloader = get_dataloader(cfg=cfg, train=True, batch_size=1, debug=True, stats=False)
    _, mask_tensor, _, mask_orig = next(iter(dataloader))  # Use mask + original mask

    pil_mask = mask_orig[0]
    if not isinstance(pil_mask, Image.Image):
        pil_mask = to_pil_image(pil_mask)

    # Resize and flip the original mask manually
    target_size = cfg["transforms"]["resize"]
    pil_resized = F.resize(pil_mask, target_size, interpolation=F.InterpolationMode.NEAREST)
    pil_flipped = F.hflip(pil_resized)
    mask_manual = F.pil_to_tensor(pil_flipped)

    mask_pipeline = mask_tensor[0]
    assert torch.equal(mask_manual, mask_pipeline), "Horizontal flip on mask did not match expected result."


def _check_resize(dataloader):
    # Get a batch of images and masks
    image_batch, mask_batch = next(iter(dataloader))
    # Expect shape: (B, C, H, W)| tests resizing
    assert isinstance(image_batch, torch.Tensor)
    assert image_batch.shape == (6, 3, 2, 2)
    assert image_batch.dtype == torch.float32

    assert isinstance(mask_batch, torch.Tensor)
    assert mask_batch.shape == (6, 1, 2, 2)
    assert mask_batch.dtype == torch.uint8

    # Expect six samples in the dataloader
    assert len(dataloader.dataset) == 6


def _check_normalize(dataloader):
    # Get all images in the dataloader
    all_images = []
    for batch_images, _ in dataloader:
        all_images.append(batch_images)
    images = torch.cat(all_images, dim=0)

    # Calculate mean and std of all images
    mean = torch.mean(images, dim=[0, 2, 3])
    std = torch.std(images, dim=[0, 2, 3])

    # Expect images to be normalized to mean around 0 and std around 1
    assert torch.allclose(mean, torch.zeros_like(mean), atol=0.5)
    assert torch.allclose(std, torch.ones_like(std), atol=0.5)

def test_train_pipeline(toy_dataset: str) -> None:
    """
    """
    calculate_mean_std()
    cfg = yaml.safe_load(open(toy_dataset))
    dataloader = get_dataloader(cfg=cfg, train=True, batch_size=6, debug=False, stats=False)

    _check_resize(dataloader)
    _check_normalize(dataloader)
    _check_color_jitter(cfg)
    _check_horizontal_flip(cfg)

# TODO: docstrings | type hints | test for val pipeline
