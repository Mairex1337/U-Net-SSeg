import os
from pathlib import Path
from typing import Any, Tuple

import numpy as np
import pytest
import src.utils.calculate_means as cm
import yaml
from PIL import Image


def make_image(path: Path, color: Tuple[int, int, int]) -> None:
    """
    Creates and saves a small RGB JPEG image at the given path with the specified color.

    Args:
        path (Path): File path where the image will be saved.
        color (Tuple[int, int, int]): RGB color values (0-255) for the entire image.
    """
    img = Image.new("RGB", (4, 4), color)
    img.save(path, format="JPEG")


def make_asymmetric_mask(path: Path) -> None:
    """
    Creates and saves a small asymmetric black-and-white PNG mask.

    This function is used for testing horizontal flipping. The left half is black (0),
    and the right half is white (255), which makes it suitable for detecting whether
    flipping was correctly applied.

    Args:
        path (Path): File path where the mask will be saved.
    """
    mask = np.full((4, 4), 0, dtype=np.uint8)
    mask[:, 2 :] = 255
    Image.fromarray(mask, mode='L').save(path)


@pytest.fixture
def toy_dataset(tmp_path: Path, monkeypatch: Any) -> str:
    """
    Pytest fixture to create a synthetic toy dataset and a temporary config file.

    It includes:
    - 6 training image-mask pairs
    - 1 validation image-mask pair
    - Each image is a solid RGB color
    - Each mask is asymmetric for flip testing
    - The transforms include resizing, normalization (with dummy values), and flip

    The fixture also monkeypatches the config loading and path resolution to ensure
    all functions operate on the temporary dataset.

    Args:
        tmp_path (Path): Temporary directory provided by pytest.
        monkeypatch (Any): Pytest fixture for modifying or overriding functions during tests.

    Returns:
        str: Path to the temporary cfg.yaml file.
    """

    data_dir = tmp_path / "data"
    train_img = data_dir / "train_images"
    train_mask = data_dir / "train_masks"
    val_img = data_dir / "val_images"
    val_mask = data_dir / "val_masks"
    for d in (train_img, train_mask, val_img, val_mask):
        d.mkdir(parents=True)

    make_image(train_img / "0.jpg", (255, 0, 0))
    make_asymmetric_mask(train_mask / "0.png")
    make_image(train_img / "1.jpg", (0, 255, 0))
    make_asymmetric_mask(train_mask / "1.png")
    make_image(train_img / "2.jpg", (0, 0, 255))
    make_asymmetric_mask(train_mask / "2.png")
    make_image(train_img / "3.jpg", (255, 0, 0))
    make_asymmetric_mask(train_mask / "3.png")
    make_image(train_img / "4.jpg", (255, 0, 0))
    make_asymmetric_mask(train_mask / "4.png")
    make_image(train_img / "5.jpg", (0, 255, 0))
    make_asymmetric_mask(train_mask / "5.png")


    make_image(val_img / "0.jpg", (0, 0, 255))
    make_asymmetric_mask(val_mask / "0.png")

    cfg: dict[str, Any] = {
        "data": {
            "train_images": str(train_img),
            "train_masks": str(train_mask),
            "val_images": str(val_img),
            "val_masks": str(val_mask),
        },
        "transforms": {
            "resize": [2, 2],
            "normalize": {"mean": [0, 0, 0], "std": [1, 1, 1]},
            "flip": 0.5,
        },
    }
    cfg_path = tmp_path / "cfg.yaml"
    with open(cfg_path, "w") as f:
        yaml.safe_dump(cfg, f)

    def fake_resolve_path(path: str, up: int) -> str:
        """
        Custom path resolver for tests to return our temporary config path.

        Args:
            path (str): The file path requested.
            up (int): How many directory levels to go up (ignored here for testing).


        Returns:
            str: The appropriate path to use for config or data access.
        """
        if os.path.basename(path) == "cfg.yaml":
            return str(cfg_path)
        return path

    monkeypatch.setenv("PYTHONPATH", str(tmp_path))
    monkeypatch.setattr(cm,
        "resolve_path",
        fake_resolve_path
    )
    monkeypatch.setattr(cm,
        "read_config",
        lambda: yaml.safe_load(cfg_path.read_text())
    )

    return str(cfg_path)
