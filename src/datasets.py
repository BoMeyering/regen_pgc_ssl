import torch
import cv2
import os
import random
import argparse
import numpy as np
import albumentations as A
from PIL import Image
from scipy.io import loadmat
import torch.utils
from torch.utils.data import Dataset, Sampler
from typing import Union, Any, Tuple
from pathlib import Path
from glob import glob

import torch.utils.data

from src.transforms import (
    get_tensor_transforms,
    get_train_transforms,
    get_val_transforms,
    get_strong_transforms,
    get_weak_transforms,
)
from logging import getLogger


class StatDataset(Dataset):
    """
    Barebones dataset implemented to iterate through all images in a single directory
    """

    def __init__(self, dir_path: Union[str, Path]):

        self.dir_path = Path(dir_path)
        self.image_keys = sorted(
            [
                img
                for img in glob("*", root_dir=self.dir_path)
                if img.endswith(("png", "jpg"))
            ]
        )
        self.transforms = get_tensor_transforms()

    def __getitem__(self, index) -> np.ndarray:
        key = self.image_keys[index]
        with open(str(self.dir_path / self.image_keys[index]), "rb") as f:
            check_chars = f.read()[-2:]
        if check_chars != b"\xff\xd9":
            try:
                img = cv2.imread(
                    str(self.dir_path / self.image_keys[index]), cv2.IMREAD_COLOR
                )
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = img.astype("float32") / 255.0  # Normalize the image to [0, 1]
                img = self.transforms(image=img)["image"]

                key = self.image_keys[index]
                return img, key
            except Exception as e:
                print(e)
                return torch.tensor(1), key
        else:
            img = cv2.imread(
                str(self.dir_path / self.image_keys[index]), cv2.IMREAD_COLOR
            )
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # img = img.astype('float32') / 255.0  # Normalize the image to [0, 1]
            img = self.transforms(image=img)["image"]

            key = self.image_keys[index]
            return img, key

    def __len__(self):
        return len(self.image_keys)


class LabelDataset(Dataset):
    """
    Barebones dataset implemented to iterate through all binary targets in a single directory
    """

    def __init__(self, dir_path: Union[str, Path]):

        self.dir_path = Path(dir_path)
        self.image_keys = sorted(
            [img for img in glob("*", root_dir=self.dir_path) if img.endswith(("png"))]
        )
        self.transforms = get_tensor_transforms()

    def __getitem__(self, index) -> np.ndarray:
        key = self.image_keys[index]
        try:
            img = cv2.imread(
                str(self.dir_path / self.image_keys[index]), cv2.IMREAD_GRAYSCALE
            )
            img = self.transforms(image=img)["image"]
            key = self.image_keys[index]
            return img, key
        except Exception as e:
            print(e)
            return torch.tensor(1), key

    def __len__(self):
        return len(self.image_keys)


class LabeledDataset(Dataset):
    def __init__(
        self, root_dir: Union[Path, str], transforms: A.Compose = get_train_transforms()
    ):
        self.img_dir = Path(root_dir) / "images"
        self.target_dir = Path(root_dir) / "labels"
        self.transforms = transforms

        if not os.path.exists(self.img_dir):
            raise NotADirectoryError(
                f"Path to img_dir {self.img_dir} does not exist. Please check path integrity."
            )
        elif not os.path.exists(self.target_dir):
            raise NotADirectoryError(
                f"Path to target_dir {self.target_dir} does not exist. Please check path integrity."
            )

        self.img_keys = sorted(
            [img for img in glob("*", root_dir=self.img_dir) if img.endswith(("jpg"))]
        )
        self.target_keys = sorted(
            [t for t in glob("*", root_dir=self.target_dir) if t.endswith(("png"))]
        )

        if len(self.img_keys) != len(self.target_keys):
            raise ValueError(
                f"Image keys and target keys are different lengths. Please ensure that each training image in {self.img_dir} has a corresponding target mask in {self.target_dir}"
            )
        elif len(self.img_keys) == 0 | len(self.target_keys) == 0:
            raise FileExistsError(
                f"Image and target directories are empty. Please ensure that the right directories were passed in the configuration file."
            )

        elif set([i.replace(".jpg", "") for i in self.img_keys]) != set(
            [i.replace(".png", "") for i in self.target_keys]
        ):
            raise ValueError(
                f"Base names for 'img_keys' and 'target_keys' do not match. Please ensure that each image in {self.img_dir} has a corresponding target mask in {self.target_dir} with the same base name."
            )

    def __getitem__(self, index) -> Any:

        # grab data keys
        img_key = self.img_keys[index]
        target_key = self.target_keys[index]

        # Paths for images and target
        img_path = Path(self.img_dir) / img_key
        target_path = Path(self.target_dir) / target_key

        # read in images and targets
        img = cv2.imread(str(img_path))
        target = cv2.imread(str(target_path), cv2.IMREAD_GRAYSCALE)

        # transform images and targets
        transformed = self.transforms(image=img, target=target)
        img, target = transformed["image"], transformed["target"]

        return img, target

    def __len__(self):

        return len(self.img_keys)


class UnlabeledDataset(Dataset):
    def __init__(
        self,
        root_dir: Union[Path, str],
        weak_transforms: A.Compose=get_weak_transforms(),
        strong_transforms: A.Compose=get_strong_transforms(),
    ):
        self.img_dir = Path(root_dir) / "images"
        self.weak_transforms = weak_transforms
        self.strong_transforms = strong_transforms

        if not os.path.exists(self.img_dir):
            raise NotADirectoryError(
                f"Path to img_dir {self.img_dir} does not exist. Please check path integrity."
            )

        self.img_keys = sorted(
            [img for img in glob("*", root_dir=self.img_dir) if img.endswith(("jpg"))]
        )

    def __getitem__(self, index) -> Any:

        # grab data keys
        img_key = self.img_keys[index]

        # Paths for images and target
        img_path = Path(self.img_dir) / img_key

        # read in images and targets
        img = cv2.imread(str(img_path))

        # transform images and targets
        weak_img = self.weak_transforms(image=img)["image"]
        # weak_array = np.moveaxis(weak_img.cpu().numpy(), source=0, destination=2)
        # strong_img = self.strong_transforms(image=weak_array)["image"]

        # return weak_img, strong_img

        return weak_img

    def __len__(self):

        return len(self.img_keys)
