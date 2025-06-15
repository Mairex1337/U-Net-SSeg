import random
from typing import Any, Callable, Dict, List, Tuple

import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from PIL import Image
from torch import Tensor


class RandomHorizontalFlip:
    """
    Randomly flip both image and mask horizontally.

    Args:
        p (float): probability of flipping.
    """
    def __init__(self, p: float=0.5) -> None:
        self.p = p
    def __call__(self, img: Image.Image, mask: Image.Image) -> Tuple[Image.Image, Image.Image]:
        """
        Horizontally flip image and mask with probability p.

        Args:
            img (PIL.Image.Image): Input image.
            mask (PIL.Image.Image): Corresponding mask.

        Returns:
            Tuple[PIL.Image.Image, PIL.Image.Image]: Transformed image and mask.
        """
        if random.random() < self.p:
            return F.hflip(img), F.hflip(mask)
        return img, mask


class ColorJitter:
    """Apply color jitter to the image only."""
    def __init__(self) -> None:
        self.color_jitter = transforms.ColorJitter(
            brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
        )

    def __call__(self, img: Image.Image, mask: Image.Image) -> Tuple[Image.Image, Image.Image]:
        """
        Apply color jitter to image only.

        Args:
            img (PIL.Image.Image): Input image.
            mask (PIL.Image.Image): Corresponding mask.

        Returns:
            Tuple[PIL.Image.Image, PIL.Image.Image]: Transformed image, and mask.
        """
        return self.color_jitter(img), mask


class Resize:
    """
    Resize both image and mask to a fixed size.

    Args:
        size (Tuple[int, int]): target (width, height).
    """
    def __init__(self, size: Tuple[int, int]) -> None:
        self.size = size

    def __call__(self, img: Image.Image, mask: Image.Image) -> Tuple[Image.Image, Image.Image]:
        """
        Resize image and mask.

        Args:
            img (PIL.Image.Image): Input image.
            mask (PIL.Image.Image): Corresponding mask.

        Returns:
            Tuple[PIL.Image.Image, PIL.Image.Image]: Resized image and mask.
        """
        img_resized  = F.resize(img,  self.size)
        mask_resized = F.resize(
            mask,
            self.size,
            interpolation=F.InterpolationMode.NEAREST
        )
        return img_resized, mask_resized


class ToTensor:
    """Convert image and mask to torch Tensors."""
    def __call__(self, img: Image.Image, mask: Image.Image) -> Tuple[Tensor, Tensor]:
        """
        Convert image to float tensor and mask to int tensor.

        Args:
            img (PIL.Image.Image): Input image.
            mask (PIL.Image.Image): Corresponding mask.

        Returns:
            Tuple[Tensor, Tensor]: Image and mask tensors.
        """
        return F.to_tensor(img), F.pil_to_tensor(mask)


class Normalize:
    """
    Normalize image tensor; mask tensor is unchanged.

    Args:
        mean (List[float]): RGB means
        std  (List[float]): RGB standard deviations
    """
    def __init__(self, mean: list[float], std: list[float]) -> None:
        self.mean = mean
        self.std = std

    def __call__(self, img: Tensor, mask: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Normalize image tensor only.

        Args:
            img (Tensor): Image tensor.
            mask (Tensor): Corresponding mask.

        Returns:
            Tuple[Tensor, Tensor]: Normalized image, and mask.
        """
        return F.normalize(img, self.mean, self.std), mask.squeeze().long()

class RandomScale:
    """
    Randomly resize both image and mask.

    Args:
        min_size (float): minimum resize
        max_size  (float): maximum resize
        og_size (list[int]): original img, mask size.
    """
    def __init__(self, min_size: float, max_size: float, og_size: list[int]) -> None:
        self.min_size = min_size
        self.max_size = max_size
        self.og_size = og_size

    def __call__(self, img: Image.Image, mask: Image.Image) -> Tuple[Image.Image, Image.Image]:
        """
        Resize both img and mask.

        Args:
            img (Image.Image): Image
            mask (Image.Image): Corresponding mask

        Returns:
            Tuple[Image.Image, Image.Image]: Randomly resized image and mask.
        """
        num = random.uniform(self.min_size, self.max_size)
        h, w = [int(round(x * num, 0)) for x in self.og_size]
        if h % 2 != 0:
            h += 1
        if w % 2 != 0:
            w += 1
        size = [h, w]
        img = F.resize(img, size)
        mask = F.resize(mask, size, interpolation=F.InterpolationMode.NEAREST)
        return img, mask
    

class PadIfSmaller:
    """
    Pad image and mask to min_size if smaller.

    Args:
        min_size (float): minimum size needed for crop
    """
    def __init__(self, min_size: list[int]) -> None:
        self.min_h, self.min_w = min_size

    def __call__(self, img: Image.Image, mask: Image.Image) -> Tuple[Image.Image, Image.Image]:
        """
        Pad image and mask.

        Args:
            img (Image.Image): Image
            mask (Image.Image): Corresponding mask

        Returns:
            Tuple[Image.Image, Image.Image]: Padded image and mask.
        """
        assert img.size == mask.size
        w, h = img.size
        h_pad = max(0, self.min_h - h)
        w_pad = max(0, self.min_w - w)

        left   = w_pad // 2
        right  = w_pad - left
        top    = h_pad // 2
        bottom = h_pad - top

        img = F.pad(img, [left, top, right, bottom], fill=255)
        mask = F.pad(mask, [left, top, right, bottom], fill=255)
        return img, mask


class RandomCrop:
    """
    Randomly crop both image and mask to goal size.

    Args:
        crop_size (tuple[int, int]): goal crop size.
    """
    def __init__(self, crop_size: tuple[int, int]) -> None:
        self.h, self.w = crop_size

    def __call__(self, img: Image.Image, mask: Image.Image) -> Tuple[Image.Image, Image.Image]:
        """
        Crop both img and mask.

        Args:
            img (Image.Image): Image
            mask (Image.Image): Corresponding mask.

        Returns:
            Tuple[Image.Image, Image.Image]: Randomly cropped image and mask.
        """
        assert img.size == mask.size
        w, h = img.size
        valid_coords = h - self.h, w - self.w
        top = random.randint(0, valid_coords[0])
        left = random.randint(0, valid_coords[1])
        return F.crop(img, top, left, self.h, self.w), F.crop(mask, top, left, self.h, self.w)

class Compose:
    """
    Composes multiple transforms to be executed in order.

    Args:
        transforms (List[Callable]) Sequence of transforms.
    """
    def __init__(self, transforms: List[Callable]) -> None:
        self.transforms = transforms

    def __call__(self, img: Image.Image, mask: Image.Image) -> Tuple[Tensor, Tensor]:
        """
        Apply sequence of transforms to image and mask.

        Args:
            img (PIL.Image.Image): Input image.
            mask (PIL.Image.Image): Corresponding mask.

        Returns:
            Tuple[Tensor, Tensor]: Transformed image and mask Tensors.
        """
        for t in self.transforms:
            img, mask = t(img, mask)
        return img, mask


def get_train_transforms(cfg: Dict[str, Any]) -> Compose:
    """
    Builds the training transform pipeline.

    Args:
        cfg: Dictionary with keys:
            - resize: Tuple[int, int]
            - flip: float
            - normalize: dict with 'mean' and 'std' lists

    Returns:
        Compose: Pipeline that takes (img, mask) and returns (img, mask).
    """
    operations = [
        RandomScale(cfg["min_scale"], cfg["max_scale"], cfg["og_scale"]),
        PadIfSmaller(cfg["resize"]),
        RandomCrop(cfg["resize"]),
        RandomHorizontalFlip(cfg["flip"]),
        ColorJitter(),
        ToTensor(),
        Normalize(cfg["normalize"]["mean"], cfg["normalize"]["std"]),
    ]
    return Compose(operations)


def get_val_transforms(cfg: Dict[str, Any]) -> Compose:
    """
    Builds the validation transform pipeline.

    Args:
        cfg: Dictionary with keys:
            - resize: Tuple[int, int]
            - flip: float
            - normalize: dict with 'mean' and 'std' lists

    Returns:
        Compose: Pipeline that takes (img, mask) and returns (img, mask).
    """
    operations = [
        Resize(cfg["resize"]),
        ToTensor(),
        Normalize(cfg["normalize"]["mean"], cfg["normalize"]["std"]),
    ]
    return Compose(operations)


def get_stats_transforms(cfg: Dict[str, Any]) -> Compose:
    """
    Builds a transform pipeline for dataset stats (mean/std) calculation.
    Applies only deterministic, value-preserving transforms (e.g., Resize, ToTensor).

    Args:
        cfg: Dictionary with keys:
            - resize: Tuple[int, int]

    Returns:
        Compose: Transform pipeline without Normalize or augmentation.
    """
    operations = [
        Resize(cfg["resize"]),
        ToTensor()
    ]
    return Compose(operations)
