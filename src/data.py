import torch
import cv2
import io
import numpy as np
import albumentations as A
from PIL import Image
from scipy.io import loadmat
from s3torchconnector import S3MapDataset, S3IterableDataset
from torch.utils.data import Dataset
from typing import Union
from pathlib import Path
from glob import glob

from src.transforms import get_tensor_transforms

# You need to update <BUCKET> and <PREFIX>
# IMG_URI="s3://kura-clover-datasets/K1702-kura-phenotyping/images/"
# TARGET_URI='s3://kura-clover-datasets/K1702-kura-phenotyping/annotations/'
# REGION = "us-east-1"

# image_dataset = S3MapDataset.from_prefix(IMG_URI, region=REGION)
# target_dataset = S3MapDataset.from_prefix(TARGET_URI, region=REGION)
# print(len(image_dataset))
# for i in range(len(image_dataset)):
#     try:
#         img_obj = image_dataset[i]
#         target_obj = target_dataset[i]

#         img_content = img_obj.read()
#         target_content = target_obj.read()

#         img = np.array(Image.open(io.BytesIO(img_content)))
#         target = loadmat(io.BytesIO(target_content))

#         print(target)

        # Read the object content into a BytesIO buffer
        # object = image_dataset[i]
        # print(object.key)
        # content = object.read()
        # img = Image.open(io.BytesIO(content))
        # img = np.array(img)
        # print(img)
        # content = io.BytesIO(object.read())
        # # Attempt to open it as an image
        # img = Image.open(content)
        # img.show()  # Optionally display the image to verify it's correct
    # except Exception as e:
    #     print(f"Failed to open image at index {i}: {e}")


class StatDataset(Dataset):
    """
    Barebones dataset implemented to iterate through all images in a directory
    """
    def __init__(self, dir_path: Union[str, Path]):

        self.dir_path = Path(dir_path)
        print(self.dir_path)
        self.image_keys = sorted([img for img in glob('*', root_dir = self.dir_path) if img.endswith(('png', 'jpg'))])
        self.transforms = get_tensor_transforms()

    def __getitem__(self, index) -> np.ndarray:
        img = cv2.imread(str(self.dir_path / self.image_keys[index])) / 255.0
        img = self.transforms(image=img)['image']

        return img
    
    def __len__(self):
        return len(self.image_keys)