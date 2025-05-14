import os
from pathlib import Path
from typing import Any, Tuple
import src.utils.calculate_means as cm


import pytest
import yaml
from PIL import Image
from src.data.dataloader import get_dataloader
from src.utils.calculate_means import calculate_mean_std
from src.utils.read_config import read_config
from torch.utils.data import DataLoader


def make_image(path: Path, color: Tuple[int, int, int]) -> None:
    """
    Create and save a tiny RGB JPEG image at the given path.
    """
    img = Image.new("RGB", (2, 2), color)
    img.save(path, format="JPEG")


def make_mask(path: Path, color: Tuple[int, int, int]) -> None:
    """
    Create and save a tiny RGB PNG mask at the given path.
    """
    mask = Image.new("RGB", (2, 2), color)
    mask.save(path, format="PNG")


@pytest.fixture
def toy_dataset(tmp_path: Path, monkeypatch: Any) -> str:
    """
    Creates a toy dataset with two training and one validation example including
    an image/mask each and a cfg.yaml pointing at them.
    Returns path to the generated cfg.yaml.
    """
    # Directories
    data_dir = tmp_path / "data"
    train_img = data_dir / "train_images"
    train_mask = data_dir / "train_masks"
    val_img = data_dir / "val_images"
    val_mask = data_dir / "val_masks"
    for d in (train_img, train_mask, val_img, val_mask):
        d.mkdir(parents=True)

    # Two train samples: red and green
    make_image(train_img / "0.jpg", (255, 0, 0))
    make_mask(train_mask / "0.png", (255, 0, 0))
    make_image(train_img / "1.jpg", (0, 255, 0))
    make_mask(train_mask / "1.png", (0, 255, 0))

    # One val sample: blue
    make_image(val_img / "0.jpg", (0, 0, 255))
    make_mask(val_mask / "0.png", (0, 0, 255))

    # Write a minimal cfg.yaml
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
            "flip": 0.0,
        },
    }
    cfg_path = tmp_path / "cfg.yaml"
    with open(cfg_path, "w") as f:
        yaml.safe_dump(cfg, f)

    def fake_resolve_path(path: str, up: int) -> str:
        # If they're asking for the config, give them the cfg.yaml
        if os.path.basename(path) == "cfg.yaml":
            return str(cfg_path)
        # Otherwise assume `path` is already an absolute or tmp_path-based data directory
        # and return it unchanged so your dataset can list its contents.
        return path

    # Monkeypatch overrides resolve_path and read_config
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


def test_calculate_mean_std_updates_cfg(toy_dataset: str) -> None:
    """
    Run calculate_mean_std and verify that our temporary cfg.yaml is updated
    with correct mean/std.
    """
    calculate_mean_std()
    cfg = yaml.safe_load(open(toy_dataset, "r"))

    mean_vals = cfg["transforms"]["normalize"]["mean"]
    std_vals = cfg["transforms"]["normalize"]["std"]

    # Expect mean = [0.5, 0.5, 0.0], std = [0.5, 0.5, 0.0]
    assert mean_vals == pytest.approx([0.5, 0.5, 0.0], rel=1e-3, abs=2e-3)
    assert std_vals == pytest.approx([0.5, 0.5, 0.0], rel=1e-3, abs=2e-3)
