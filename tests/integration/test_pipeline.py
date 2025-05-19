from typing import Any, Dict

import torch
import yaml
from PIL import Image
from scripts.calculate_means import calculate_mean_std
from src.data.dataloader import get_dataloader
from tests.integration.utils.toy_dataset import toy_dataset
from torch.utils.data import DataLoader
from torchvision.transforms import functional as F
from torchvision.transforms.functional import to_pil_image


def _check_color_jitter(cfg: Dict[str, Any]) -> None:
    """
    Verifies that color jittering is applied during training data augmentation.

    This test disables horizontal flipping to isolate the effects of ColorJitter.
    It fetches two consecutive batches from the DataLoader and asserts that their
    pixel values differ, indicating that random jitter was applied.

    Args:
        cfg (Dict[str, Any]): Configuration dictionary containing the transform parameters.

    Raises:
        AssertionError: If the difference between two image batches is too small,
                        indicating that ColorJitter may not be applied.
    """
    cfg["transforms"]["flip"] = 0.0
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

    target_size = cfg["transforms"]["resize"]
    pil_resized = F.resize(pil_mask, target_size, interpolation=F.InterpolationMode.NEAREST)
    pil_flipped = F.hflip(pil_resized)
    mask_manual = F.pil_to_tensor(pil_flipped).squeeze(0)

    mask_pipeline = mask_tensor[0]
    assert torch.equal(mask_manual, mask_pipeline), "Horizontal flip on mask did not match expected result."


def _check_resize(dataloader: DataLoader) -> None:
    """
    Verifies that all images and masks are resized to the expected dimensions.

    This function checks that the shape and datatype of the image and mask tensors
    match the expected format after resizing during the transform pipeline.

    Args:
        dataloader (DataLoader): The DataLoader providing transformed batches.

    Raises:
        AssertionError: If image or mask tensors are not correctly resized or typed,
                        or if the dataset does not contain the expected number of samples.
    """
    image_batch, mask_batch = next(iter(dataloader))

    assert isinstance(image_batch, torch.Tensor)
    assert image_batch.shape == (6, 3, 2, 2), "Image batch is not resized to expected shape."
    assert image_batch.dtype == torch.float32

    assert isinstance(mask_batch, torch.Tensor)
    assert mask_batch.shape == (6, 2, 2), "Mask batch is not resized to expected shape."
    assert mask_batch.dtype == torch.int64

    assert len(dataloader.dataset) == 6, "Dataset does not contain 6 samples."


def _check_normalize(dataloader: DataLoader) -> None:
    """
    Verifies that the normalization transform was applied correctly.

    This function computes the mean and standard deviation of the image batch and
    asserts that they are approximately zero and one, respectively, indicating that
    normalization has been applied using calculated values.

    Args:
        dataloader (DataLoader): The DataLoader providing normalized images.

    Raises:
        AssertionError: If the mean or std dev is not within tolerance of expected values.
    """
    all_images = []
    for batch_images, _ in dataloader:
        all_images.append(batch_images)
    images = torch.cat(all_images, dim=0)

    mean = torch.mean(images, dim=[0, 2, 3])
    std = torch.std(images, dim=[0, 2, 3])

    assert torch.allclose(mean, torch.zeros_like(mean), atol=0.5), f"Unexpected mean: {mean}"
    assert torch.allclose(std, torch.ones_like(std), atol=0.5), f"Unexpected std: {std}"


def test_train_pipeline(toy_dataset: str) -> None:
    """
    Integration test that validates the full data preprocessing pipeline.

    This test runs the full transformation pipeline, including resizing,
    normalization, color jitter, and horizontal flipping. Each step is validated
    individually to ensure that data augmentation and preprocessing are
    functioning as expected.

    Args:
        toy_dataset (str): Path to a temporary configuration file pointing to toy dataset.

    Raises:
        AssertionError: If any preprocessing step does not produce the expected result.
    """
    calculate_mean_std()
    cfg = yaml.safe_load(open(toy_dataset))
    dataloader = get_dataloader(cfg=cfg, train=True, batch_size=6, debug=False, stats=False)

    _check_resize(dataloader)
    _check_normalize(dataloader)
    _check_color_jitter(cfg)
    _check_horizontal_flip(cfg)
