# Image and target transformations
# BoMeyering 2024

import json
import albumentations
import albumentations as A
from albumentations.pytorch import ToTensorV2
from typing import Tuple, Iterable, Union

# Grab the defaults
with open('metadata/dataset_norm.json', 'r') as f:
    norms = json.load(f)
means = norms.get('means')
std = norms.get('std')

def get_train_transforms(resize: Tuple=(512, 512), means: Iterable=means, std: Iterable=std) -> albumentations.Compose:
    """
    Return a training transformation for the training images and targets

    Args:
        resize (Tuple, optional): Tuple of new output height and width. Defaults to (512, 512).
        means (Iterable, optional): Tuple of the the RGB channel means to normalize by. Defaults to means.
        std (Iterable, optional): Tuple of the RGB channel standard deviations to normalize by. Defaults to std.

    Returns:
        albumentations.Compose: A Compose function to use in the datasets. 
    """
    transforms = A.Compose([
        A.Resize(*resize, p=1),
        A.Affine(),
        A.SafeRotate(),
        A.HorizontalFlip(),
        A.GaussianBlur(),
        A.Normalize(mean=means, std=std),
        ToTensorV2()
    ], additional_targets={'target': 'mask'})

    return transforms

def get_strong_transforms(resize: Tuple=(512, 512), means: Iterable=means, std: Iterable=std) -> albumentations.Compose:
    """
    Return a strong training transformation for the unlabeled training images and targets

    Args:
        resize (Tuple, optional): Tuple of new output height and width. Defaults to (512, 512).
        means (Iterable, optional): Tuple of the the RGB channel means to normalize by. Defaults to means.
        std (Iterable, optional): Tuple of the RGB channel standard deviations to normalize by. Defaults to std.

    Returns:
        albumentations.Compose: A Compose function to use in the datasets. 
    """

    print
    transforms = A.Compose([
        A.Resize(*resize, p=1),
        A.Affine(),
        A.SafeRotate(),
        A.HorizontalFlip(),
        A.GaussianBlur(),
        A.Solarize(),
        A.ChannelShuffle(),
        A.RGBShift(),
        A.Normalize(mean=means, std=std),
        ToTensorV2()
    ], additional_targets={'target': 'mask', 'conf_mask': 'mask'})

    return transforms

def get_weak_transforms(resize: Tuple=(512, 512), means: Iterable=means, std: Iterable=std) -> albumentations.Compose:
    """
    Return a weak training transformation for the unlabeled training images and targets

    Args:
        resize (Tuple, optional): Tuple of new output height and width. Defaults to (512, 512).
        means (Iterable, optional): Tuple of the the RGB channel means to normalize by. Defaults to means.
        std (Iterable, optional): Tuple of the RGB channel standard deviations to normalize by. Defaults to std.

    Returns:
        albumentations.Compose: A Compose function to use in the datasets. 
    """
    transforms = A.Compose([
        A.Resize(*resize, p=1),
        A.Affine(),
        A.GaussianBlur(),
        A.HorizontalFlip(),
        A.Normalize(mean=means, std=std),
        ToTensorV2()
    ])

    return transforms

def get_val_transforms(resize: Tuple=(512, 512), means: Iterable=means, std: Iterable=std) -> albumentations.Compose:
    """
    Return a transform function for validation transforms, i.e. just resize and normalize.

    Args:
        resize (Tuple, optional): Tuple of new output height and width. Defaults to (512, 512).
        means (Iterable, optional): Tuple of the the RGB channel means to normalize by. Defaults to means.
        std (Iterable, optional): Tuple of the RGB channel standard deviations to normalize by. Defaults to std.

    Returns:
        albumentations.Compose: A Compose function to use in the datasets. 
    """
    transforms = A.Compose([
        A.Resize(*resize, p=1),
        A.Normalize(mean=means, std=std),
        ToTensorV2()
    ], additional_targets={'target': 'mask'})

    return transforms

def get_null_transforms(resize: Tuple=(512, 512), means: Iterable=means, std: Iterable=std) -> albumentations.Compose:
    """
    Return a transform function for null transforms, i.e. just resize and normalize

    Args:
        resize (Tuple, optional): Tuple of new output height and width. Defaults to (512, 512).
        means (Iterable, optional): Tuple of the the RGB channel means to normalize by. Defaults to means.
        std (Iterable, optional): Tuple of the RGB channel standard deviations to normalize by. Defaults to std.

    Returns:
        albumentations.Compose: A Compose function to use in the datasets. 
    """
    transforms = A.Compose([
        A.Resize(*resize, p=1),
        A.Normalize(mean=means, std=std),
        ToTensorV2()
    ], additional_targets={'target': 'mask'})

    return transforms

def get_tensor_transforms(resize: Union[Tuple, None] = (512, 512)) -> albumentations.Compose:
    """
    Return a tensor transformation for the testing purposes.

    Returns:
        albumentations.Compose: A Compose function to use in the datasets. 
    """
    if resize:
        resize_p = 0
    else:
        resize_p = 1

    transforms = A.Compose([
        A.Resize(*resize, p=resize_p),
        ToTensorV2(p=1.0)
    ], additional_targets={'target': 'mask'})
 
    return transforms