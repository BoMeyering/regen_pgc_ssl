# Class distribution calculation
# BoMeyering 2024

import torch
import os
import sys
import json
import pandas as pd
from glob import glob
from tqdm import tqdm
from torch.utils.data import DataLoader

script_directory = os.path.dirname(os.path.abspath(__file__))
root_directory = os.path.dirname(script_directory)
sys.path.append(root_directory)

from src.datasets import LabelDataset

class ClassMeters:
    def __init__(self):
        self.pixel_meters = {
            '0': 0,
            '1': 0,
            '2': 0,
            '3': 0,
            '4': 0,
            '5': 0,
            '6': 0,
            '7': 0
        }

        self.img_meters = {
            '0': 0,
            '1': 0,
            '2': 0,
            '3': 0,
            '4': 0,
            '5': 0,
            '6': 0,
            '7': 0
        }

    def update_img(self, idx: int):
        self.img_meters[str(idx)] += 1
    
    def update_px(self, idx: int, pixels: int):
        self.pixel_meters[str(idx)] += pixels

    def display(self):
        print(self.img_meters)
        print(self.pixel_meters)


def main():

    label_dir = 'data/processed/labeled/train/labels'
    # target_files = glob('*.png', root_dir=label_dir)

    dataset = LabelDataset(dir_path=label_dir)
    print(len(dataset))
    label_dl = DataLoader(
        dataset=dataset, 
        batch_size=1, 
        num_workers=2
    )

    iter_loader = iter(label_dl)

    img_keys = []
    pixel_array = []
    meters = ClassMeters()
    for _ in tqdm(range(len(dataset))):
        img, key = next(iter_loader)
        
        pixel_list = [0, 0, 0, 0, 0, 0, 0, 0]
        unique_labels = torch.unique(img)
        for i in unique_labels:
            i = i.item()
            meters.update_img(idx=i)
            pixels = (img == i).sum().item()
            meters.update_px(idx=i, pixels=pixels)
            pixel_list[i] = pixels
        # meters.display()
        pixel_array.append(pixel_list)
        img_keys.append(key)
    with open('metadata/class_img_counts.json', 'w') as f:
        json.dump(meters.img_meters, f)
    with open('metadata/class_pixel_counts.json', 'w') as f:
        json.dump(meters.pixel_meters, f)
    
    pixel_dist_df = pd.DataFrame(pixel_array)
    pixel_dist_df['keys'] = img_keys

    pixel_dist_df.to_csv('metadata/img_pixel_distributions.csv')
        


if __name__ == '__main__': 
    main()