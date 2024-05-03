# Image and target transformations
# BoMeyering 2024

import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_train_transforms(args):
    
    transforms = A.Compose([
        A.Affine(),
        A.SafeRotate(),
        A.HorizontalFlip(),
        A.GaussianBlur(),
        A.Solarize(),
        A.ChannelShuffle(),
        A.RGBShift(),
        A.Normalize(),
        ToTensorV2()
    ], additional_targets={'target': 'mask'})

    return transforms

def get_strong_transforms(args):

    transforms = A.Compose([
        A.Affine(),
        A.SafeRotate(),
        A.HorizontalFlip(),
        A.GaussianBlur(),
        A.Solarize(),
        A.ChannelShuffle(),
        A.RGBShift(),
        A.Normalize(),
        ToTensorV2()
    ], additional_targets={'target': 'mask'})

    return transforms

def get_weak_transforms(args):

    transforms = A.Compose([
        A.Affine(),
        A.GaussianBlur(),
        A.HorizontalFlip(),
        A.Normalize(),
        ToTensorV2()
    ], additional_targets={'target': 'mask'})

    return transforms

def get_val_transforms(args):
    transforms = A.Compose([
        A.Normalize(),
        ToTensorV2()
    ], additional_targets={'target': 'mask'})

    return transforms

def get_null_transforms():
    transforms = A.Compose([
        A.Normalize(),
        ToTensorV2()
    ])

    return transforms

def get_tensor_transforms():
    transforms = A.Compose([
        ToTensorV2(p=1.0)
    ])

    return transforms

    return True