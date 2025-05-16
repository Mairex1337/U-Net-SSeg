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
    Create and save a tiny RGB JPEG image at the given path.
    """
    img = Image.new("RGB", (4, 4), color)
    img.save(path, format="JPEG")


def make_asymmetric_mask(path: Path) -> None:
    """
    Creates an asymmetrical black and white mask and saves it as PNG.
    These masks are used to test the horizontal flipping of the pipeline,
    therfore they are not symmetric.
    """
    mask = np.full((4, 4), 0, dtype=np.uint8)
    mask[:, 2 :] = 255
    Image.fromarray(mask, mode='L').save(path)


@pytest.fixture
def toy_dataset(tmp_path: Path, monkeypatch: Any) -> str:
    """
    Creates a toy dataset with two training and one validation example including
    an image/mask each and a cfg.yaml pointing at them.
    Returns path to the generated cfg.yaml.
    """
    data_dir = tmp_path / "data"
    train_img = data_dir / "train_images"
    train_mask = data_dir / "train_masks"
    val_img = data_dir / "val_images"
    val_mask = data_dir / "val_masks"
    for d in (train_img, train_mask, val_img, val_mask):
        d.mkdir(parents=True)

    # Two train samples: red and green
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


    # One val sample: blue
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
