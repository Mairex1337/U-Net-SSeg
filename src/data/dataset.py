import os

import torch.utils.data as data
from PIL import Image
from src.data.transforms import Compose
from src.utils import resolve_path
from torch import Tensor
from torchvision.transforms.functional import pil_to_tensor, to_tensor


class SegmentationDataset(data.Dataset):
    """
    A PyTorch Dataset for semantic segmentation tasks.

    Loads images and masks from, applies transforms,
    and returns them as tensors.

    Args:
        img_dir (str): Relative path to image directory.
        mask_dir (str): Relative path to mask directory.
        transforms (Compose): Sequence of transforms applied to image-mask pairs.
        debug (bool): Flag to indicate debug mode in which __getitem__ also returns
            the original image and mask without transformations.

    Raises:
        AssertionError: If image and mask filenames do not all match.
    """
    def __init__(
            self,
            img_dir: str,
            mask_dir: str,
            transforms: Compose,
            debug: bool = False
    ) -> None:
        self.transforms = transforms
        self.img_dir = resolve_path(img_dir)
        self.mask_dir = resolve_path(mask_dir)
        self.debug = debug
        self.img_idx = sorted([
            os.path.splitext(f)[0] for f in os.listdir(self.img_dir)
        ])
        self.mask_idx = sorted([
            os.path.splitext(f)[0] for f in os.listdir(self.mask_dir)
        ])
        assert self.img_idx == self.mask_idx, "Image and Mask names do not match."

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        """
        Load and return transformed image and mask at given index.

        If self.debug == True, also returns original image and mask.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            tuple[Tensor, Tensor]: Transformed image and mask tensors.
        """
        img_path = os.path.join(self.img_dir, f'{self.img_idx[idx]}.jpg')
        mask_path = os.path.join(self.mask_dir, f'{self.img_idx[idx]}.png')
        img = Image.open(img_path)
        mask = Image.open(mask_path)
        if self.transforms:
            img_t, mask_t = self.transforms(img, mask)
        else:
            img_t  = img
            mask_t = mask
        if self.debug:
            return img_t, mask_t, to_tensor(img), pil_to_tensor(mask)
        return img_t, mask_t

    def __len__(self) -> int:
        """
        Return the number of samples in the dataset.

        Returns:
            int: Number of samples in the dataset.
        """
        return len(self.img_idx)
