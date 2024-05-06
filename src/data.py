import torch
import cv2
import os
import numpy as np
import albumentations as A
from PIL import Image
from scipy.io import loadmat
from torch.utils.data import Dataset
from typing import Union, Any
from pathlib import Path
from glob import glob

from src.transforms import get_tensor_transforms
from logging import getLogger


class StatDataset(Dataset):
    """
    Barebones dataset implemented to iterate through all images in a single directory
    """
    def __init__(self, dir_path: Union[str, Path]):

        self.dir_path = Path(dir_path)
        self.image_keys = sorted([img for img in glob('*', root_dir = self.dir_path) if img.endswith(('png', 'jpg'))])
        self.transforms = get_tensor_transforms()

    def __getitem__(self, index) -> np.ndarray:
        key = self.image_keys[index]
        with open(str(self.dir_path / self.image_keys[index]), 'rb') as f:
            check_chars = f.read()[-2:]
        if check_chars != b'\xff\xd9':
            try: 
                img = cv2.imread(str(self.dir_path / self.image_keys[index]), cv2.IMREAD_COLOR)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = img.astype('float32') / 255.0  # Normalize the image to [0, 1]
                img = self.transforms(image=img)['image']

                key = self.image_keys[index]
                return img, key 
            except Exception as e:
                print(e)
                return torch.tensor(1), key
        else:
            img = cv2.imread(str(self.dir_path / self.image_keys[index]), cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img.astype('float32') / 255.0  # Normalize the image to [0, 1]
            img = self.transforms(image=img)['image']

            key = self.image_keys[index]
            return img, key 
    
    def __len__(self):
        return len(self.image_keys)
    
class LabeledDataset(Dataset):
    def __init__(self, img_dir: Union[Path, str], target_dir: Union[Path, str], transforms: A.Compose):
        self.img_dir = img_dir
        self.target_dir = target_dir
        self.transforms = transforms

        if not os.path.exists(self.img_dir):
            raise NotADirectoryError(f'Path to img_dir {self.img_dir} does not exist. Please check path integrity.')
        elif not os.path.exists(self.target_dir):
            raise NotADirectoryError(f'Path to target_dir {self.target_dir} does not exist. Please check path integrity.')

        self.img_keys = sorted([img for img in glob('*', root_dir = self.img_dir) if img.endswith(('jpg'))])
        self.target_keys = sorted([t for t in glob('*', root_dir = self.target_dir) if t.endswith(('png'))])

        if len(self.img_keys) != len(self.target_keys):
            raise ValueError(f"Image keys and target keys are different lengths. Please ensure that each training image in {self.img_dir} has a corresponding target mask in {self.target_dir}")
        elif set(self.img_keys.replace('.jpg', '')) != set(self.target_keys.replace('.png', '')):
            raise ValueError(f"Base names for 'img_keys' and 'target_keys' do not match. Please ensure that each image in {self.img_dir} has a corresponding target mask in {self.target_dir} with the same base name.")

        
        

        
    def __getitem__(self, index) -> Any:

        img_key = self.img_keys[index]
        target_key = 




        return img, target
    

    def __len__(self):

        return len(self.img)
    
