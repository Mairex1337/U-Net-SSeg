from typing import Any, Dict

import pytest
import torch
import yaml
from PIL import Image
from scripts.calculate_means import calculate_mean_std
from src.data.dataloader import get_dataloader
from tests.utils import get_batch, patch_all_except, toy_dataset
from torchvision.transforms import functional as F
from torchvision.transforms.functional import to_pil_image


@pytest.fixture(scope="session")
def calculate_means_once():
    """Fixture that calculates dataset-wide mean and std only once per test session."""
    calculate_mean_std()

@pytest.fixture
def cfg(toy_dataset: str, calculate_means_once) -> Dict[str, Any]:
    """Loads and returns the test configuration from the toy dataset YAML.

    Args:
        toy_dataset (str): Path to the toy dataset YAML file.
        calculate_means_once: Fixture to ensure mean/std are computed once.

    Returns:
        Dict[str, Any]: Configuration dictionary.
    """
    return yaml.safe_load(open(toy_dataset))


def test_resize_applied(cfg: Dict[str, Any]) -> None:
    """Checks whether images and masks are resized to the expected dimensions.

    Args:
        cfg (Dict[str, Any]): Configuration dictionary with transform settings.

    Raises:
        AssertionError: If dimensions or types do not match expectations.
    """
    image_batch, mask_batch = get_batch(cfg, batch_size=6, debug=False, stats=True)

    assert isinstance(image_batch, torch.Tensor)
    assert image_batch.shape == (6, 3, 2, 2), f"Expected image shape (6, 3, 2, 2), got {image_batch.shape}"
    assert image_batch.dtype == torch.float32

    assert isinstance(mask_batch, torch.Tensor)
    assert mask_batch.shape == (6, 1, 2, 2), f"Expected mask shape (6, 1, 2, 2), got {mask_batch.shape}"
    assert mask_batch.dtype == torch.uint8

    assert len(get_dataloader(cfg, split='train', batch_size=6).dataset) == 6, "Dataset does not contain 6 samples."


@patch_all_except("Normalize")
def test_normalization_applied(cfg: Dict[str, Any]) -> None:
    """Checks if images are normalized to approximately zero mean and unit std.

    Args:
        cfg (Dict[str, Any]): Configuration dictionary.

    Raises:
        AssertionError: If computed mean or std deviate significantly.
    """
    image_batches = []
    for batch_images, _ in get_dataloader(cfg, split='train', batch_size=6):
        image_batches.append(batch_images)
    images = torch.cat(image_batches, dim=0)

    mean = torch.mean(images, dim=[0, 2, 3])
    std = torch.std(images, dim=[0, 2, 3])

    assert torch.allclose(mean, torch.zeros_like(mean), atol=0.75), f"Unexpected mean: {mean}"
    assert torch.allclose(std, torch.ones_like(std), atol=0.75), f"Unexpected std: {std}"


@patch_all_except("ColorJitter")
def test_color_jitter_applied(cfg: Dict[str, Any]) -> None:
    """Checks whether ColorJitter produces different outputs on consecutive calls.

    Args:
        cfg (Dict[str, Any]): Configuration dictionary.

    Raises:
        AssertionError: If image difference is too small (jitter not applied).
    """
    dataloader = get_dataloader(cfg, split='train' ,batch_size=1, debug=False, stats=False)
    img1, _ = next(iter(dataloader))
    img2, _ = next(iter(dataloader))

    diff = torch.abs(img1 - img2).mean().item()
    assert diff > 0.01, f"Color jitter did not significantly change image. Mean difference: {diff:.5f}"


@patch_all_except("RandomHorizontalFlip")
def test_horizontal_flip_applied(cfg: Dict[str, Any]) -> None:
    """Checks that horizontal flipping is correctly applied to the mask.

    Args:
        cfg (Dict[str, Any]): Configuration dictionary with 'flip' probability set to 1.

    Raises:
        AssertionError: If flipped mask doesn't match manual flip result.
    """
    cfg["transforms"].update({"flip": 1.0})
    _, mask_tensor, _, mask_orig, _ = get_batch(cfg)

    pil_mask = mask_orig[0] if isinstance(mask_orig[0], Image.Image) else to_pil_image(mask_orig[0])
    pil_flipped = F.hflip(pil_mask)
    mask_manual = F.pil_to_tensor(pil_flipped).squeeze(0)

    mask_pipeline = mask_tensor[0]
    assert torch.equal(mask_manual, mask_pipeline), "Horizontal flip on mask did not match expected result."


@patch_all_except("RandomScale")
def test_random_scale_applied(cfg: Dict[str, Any]) -> None:
    """Checks that RandomScale applies varying scale factors across samples.

    Args:
        cfg (Dict[str, Any]): Configuration dictionary with scale range.

    Raises:
        AssertionError: If all scale factors are the same (indicating no scaling).
    """
    cfg["transforms"].update({"min_scale": 0.5, "max_scale": 1.5})
    scales = set()
    for _ in range(5):
        img, _, img_orig, _, _ = get_batch(cfg)
        img, img_orig = img[0], img_orig[0]
        h_new, _ = img.shape[1:]
        h_orig, _ = img_orig.shape[1:]
        scales.add(round(h_new / h_orig, 2))

    assert len(scales) > 1, f"RandomScale not applied — all scale factors are the same: {scales}"


@patch_all_except("PadIfSmaller")
def test_pad_if_smaller_applied(cfg: Dict[str, Any]) -> None:
    """Checks that PadIfSmaller resizes smaller inputs to at least the target size.

    Args:
        cfg (Dict[str, Any]): Configuration dictionary with 'resize' field.

    Raises:
        AssertionError: If the resulting size is below the target.
    """
    cfg["transforms"].update({"resize": [5, 5]})
    img, mask, *_ = get_batch(cfg)
    assert img.shape[2:] == torch.Size([5, 5]), f"Expected image size [5, 5], got {img.shape[2:]}"
    assert mask.shape[1:] == torch.Size([5, 5]), f"Expected mask size [5, 5], got {mask.shape[1:]}"


@patch_all_except("RandomCrop")
def test_random_crop_applied(cfg: Dict[str, Any]) -> None:
    """Checks that RandomCrop crops the input to the exact target size.

    Args:
        cfg (Dict[str, Any]): Configuration dictionary with resize and scale set to 1.0.

    Raises:
        AssertionError: If cropped output does not match expected size.
    """
    cfg["transforms"].update({"resize": [2, 2], "min_scale": 1.0, "max_scale": 1.0})
    img, mask, *_ = get_batch(cfg)
    assert img.shape[2:] == torch.Size([2, 2]), f"Expected image size [2, 2], got {img.shape[2:]}"
    assert mask.shape[1:] == torch.Size([2, 2]), f"Expected mask size [2, 2], got {mask.shape[1:]}"
