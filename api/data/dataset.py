import os

import torch.utils.data as data
from PIL import Image
from torch import Tensor
from torchvision import transforms


class InferenceDataset(data.Dataset):
    """
    A PyTorch Dataset used for inference.

    Loads images, applies transforms,
    and returns them as tensors.

    Args:
        img_dir (str): Absolute path to image directory.
        transforms (Compose): Sequence of transforms applied to images
    """
    def __init__(
            self,
            img_dir: str,
            transforms: transforms.Compose,
    ) -> None:
        self.transforms = transforms
        self.img_dir = img_dir
        self.img_idx = sorted([
            os.path.splitext(f)[0] for f in os.listdir(self.img_dir)
        ])

    def __getitem__(self, idx: int) -> Tensor:
        """
        Load and return transformed image at given index.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            Tensor: Transformed image tensors.
        """
        img_path = os.path.join(self.img_dir, f'{self.img_idx[idx]}.jpg')
        img = Image.open(img_path)
        if self.transforms:
            img_t = self.transforms(img)
        return img_t

    def __len__(self) -> int:
        """
        Return the number of samples in the dataset.

        Returns:
            int: Number of samples in the dataset.
        """
        return len(self.img_idx)
