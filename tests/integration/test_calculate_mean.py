import pytest
import yaml
from src.utils.calculate_means import calculate_mean_std
from tests.integration.utils.toy_dataset import toy_dataset


def test_calculate_mean_std_updates_cfg(toy_dataset: str) -> None:
    """
    Integration test that verifies whether `calculate_mean_std()` correctly updates
    the normalization values (mean and std) in the temporary configuration file.

    This test assumes a toy dataset with known pixel values across six images,
    where the expected normalized mean and standard deviation are deterministic.

    Args:
        toy_dataset (str): Path to the temporary cfg.yaml file created by the fixture.

    Asserts:
        - That the calculated per-channel means and standard deviations are approximately
          equal to the expected values within a small tolerance.
    """
    calculate_mean_std()
    cfg = yaml.safe_load(open(toy_dataset, "r"))

    mean_vals = cfg["transforms"]["normalize"]["mean"]
    std_vals = cfg["transforms"]["normalize"]["std"]

    # Expect mean = [0.5, 0.3333, 0.1667], std = [0.5, 0.4714, 0.3727]
    assert mean_vals == pytest.approx([0.5, 0.3333, 0.1667], rel=1e-3, abs=3e-3)
    assert std_vals == pytest.approx([0.5, 0.4714, 0.3727], rel=1e-3, abs=3e-3)
