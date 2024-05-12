# Dataset Remapping Script
# BoMeyering 2024

import json
import os
import shutil
import cv2
from tqdm import tqdm
import numpy as np
from glob import glob

from scipy.io import loadmat


# Process K1702 images
mapping = {
    0: ('background', 0),
    1: ('quadrat', 1),
    2: ('soil', 0),
    3: ('kura_clover', 3)
}

img_dir = "data/raw/k1702_dataset/images"
ann_dir = "data/raw/k1702_dataset/annotations"
map_path = "metadata/k1702_mapping.json"
img_out_dir = "data/processed/labeled/images"
ann_out_dir = "data/processed/labeled/labels"

ann_keys = glob('*.mat', root_dir=ann_dir)


print(mapping)
for k in tqdm(ann_keys):
    base_id = k.replace('_mask.mat', '')
    path = os.path.join(ann_dir, k)
    target = loadmat(path)['data']
    new_target = np.zeros(target.shape)
    for k, v in mapping.items():
        idx = np.where(target == k)
        new_target[idx] = v[1]
    out_path = os.path.join(ann_out_dir, base_id + '.png')
    cv2.imwrite(out_path, new_target)
    img_src = os.path.join(img_dir, base_id + '.jpg')
    img_dst = os.path.join(img_out_dir, base_id + '.jpg')
    shutil.copyfile(img_src, img_dst)

# Process grass_clover
mapping = {
    0: ('soil', 0),
    1: ('clover', 1),
    2: ('grass', 2),
    3: ('weeds', 4),
    4: ('white_clover', 3),
    5: ('red_clover', 3),
    6: ('dandelion', 4),
    7: ('shepherds_purse', 4),
    8: ('thistle', 4),
    9: ('white_clover_flower', 3),
    10:('white_clover_leaf', 3), 
    11: ('red_clover_flower', 3),
    12: ('red_clover_leaf', 3),
    13: ('unknown_clover_leaf', 3),
    14: ('unknown_clover_flower', 3)
}

img_dir = "data/raw/grass-clover-dataset/synthetic_images/synthetic_images/Images"
ann_dir = "data/raw/grass-clover-dataset/synthetic_images/synthetic_images/ImageLabels"
map_path = "metadata/grassclover_mapping.json"
img_out_dir = "data/processed/labeled/images"
ann_out_dir = "data/processed/labeled/labels"

ann_keys = glob('*.png', root_dir=ann_dir)

print(mapping)
for k in tqdm(ann_keys):
    base_id = k.replace('.png', '')
    path = os.path.join(ann_dir, k)
    target = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    new_target = np.zeros(target.shape)
    for k, v in mapping.items():
        idx = np.where(target == k)
        new_target[idx] = v[1]
    out_path = os.path.join(ann_out_dir, base_id + '.png')
    cv2.imwrite(out_path, new_target)
    img_src = os.path.join(img_dir, base_id + '.jpg')
    img_dst = os.path.join(img_out_dir, base_id + '.jpg')
    shutil.copyfile(img_src, img_dst)


# Process cropandweed dataset

mapping = {
    0: ('Soil', 0),
    1: ('Maize', 5),
    2: ('Maize two-leaf stage', 5),
    3: ('Maize four-leaf stage', 5),
    4: ('Maize six-leaf stage', 5),
    5: ('Maize eight-leaf stage', 5),
    6: ('Maize max', 5),
    7: ('Sugar beet', 7),
    8: ('Sugar beet two-leaf stage', 7),
    9: ('Sugar beet four-leaf stage', 7),
    10: ('Sugar beet six-leaf stage', 7),
    11: ('Sugar beet eight-leaf stage', 7),
    12: ('Sugar beet Max', 7),
    13: ('Pea', 7),
    14: ('Courgette', 7),
    15: ('Pumpkins', 7),
    16: ('Radish', 7),
    17: ('Asparagus', 7),
    18: ('Potato', 7),
    19: ('Flat leaf parsley', 7),
    20: ('Curly leaf parsley', 7),
    21: ('Cowslip', 4),
    22: ('Poppy', 4),
    23: ('Hemp', 4),
    24: ('Sunflower', 4),
    25: ('Sage', 4),
    26: ('Common bean',7),
    27: ('Faba bean', 7),
    28: ('Clover', 3),
    29: ('Hybrid goosefoot', 4),
    30: ('Black-bindweed', 4),
    31: ('Cockspur grass', 2),
    32: ('Red-root amaranth', 4),
    33: ('White goosefoot', 4),
    34: ('Thorn apple', 4),
    35: ('Potato weed', 4),
    36: ('German chamomile', 4),
    37: ('Saltbush', 4),
    38: ('Creeping thistle', 4),
    39: ('Field milk thistle', 4),
    40: ('Purslane', 4),
    41: ('Black nightshade', 4),
    42: ('Mercuries', 4),
    43: ('Spurge', 4),
    44: ('Pale persicaria', 4),
    45: ('Geraniums', 4),
    46: ('Cleavers', 4),
    47: ('Whitetop', 4),
    48: ('Meadow-grass', 2),
    49: ('Frosted orach', 4),
    50: ('Black horehound', 4),
    51: ('Shepherds purse', 4),
    52: ('Field bindweed', 4),
    53: ('Common mugwort', 4),
    54: ('Hedge mustard', 4),
    55: ('Groundsel', 4),
    56: ('Speedwell', 4),
    57: ('Broadleaf plantain', 4),
    58: ('White ball-mustard', 4),
    59: ('Peppermint', 4),
    60: ('Field pennycress', 4),
    61: ('Corn spurry', 4),
    62: ('Purple crabgrass', 2),
    63: ('Common fumitory', 4),
    64: ('Ivy-leaved speedwell', 4),
    65: ('Annual meadow grass', 2),
    66: ('Redshank', 4),
    67: ('Common hemp-nettle', 4),
    68: ('Rough meadow-grass', 2),
    69: ('Green bristlegrass', 2),
    70: ('Small geranium', 4),
    71: ('Cornflower', 4),
    72: ('Common corn-cockle', 4),
    73: ('Creeping crowfoot', 4),
    74: ('Wall barley', 2),
    75: ('Annual fescue', 2),
    76: ('Purple dead-nettle', 4),
    77: ('Ribwort plantain', 4),
    78: ('Pineappleweed', 4),
    79: ('Common chickweed', 4),
    80: ('Hedge mustard', 4),
    81: ('Soft brome', 2),
    82: ('Wild pansy', 4),
    83: ('Yellow rocket', 4),
    84: ('Common wild oat', 2),
    85: ('Red poppy', 4),
    86: ('Rye brome', 2),
    87: ('Knotgrass', 2),
    88: ('Prickly lettuce', 4),
    89: ('Copse-bindweed', 4),
    90: ('Manyseeds', 4),
    91: ('Common buckwheat', 4),
    92: ('Chives', 4),
    93: ('Garlic', 4),
    94: ('Soybean', 6),
    95: ('Wild carrot', 4),
    96: ('Field mustard', 4),
    97: ('Giant fennel',4),
    98: ('Common horsetail', 4),
    99: ('Common dandelion', 4),
    255: ('Vegetation', 7)}


img_dir = "data/raw/cropandweed-dataset/data/images"
ann_dir = "data/raw/cropandweed-dataset/data/labelIds/CropAndWeed"
img_out_dir = "data/processed/labeled/images"
ann_out_dir = "data/processed/labeled/labels"

ann_keys = glob('*.png', root_dir=ann_dir)

print(mapping)
for k in tqdm(ann_keys):
    base_id = k.replace('.png', '')
    path = os.path.join(ann_dir, k)
    target = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    new_target = np.zeros(target.shape)
    for k, v in mapping.items():
        idx = np.where(target == k)
        new_target[idx] = v[1]
    out_path = os.path.join(ann_out_dir, base_id + '.png')
    cv2.imwrite(out_path, new_target)
    img_src = os.path.join(img_dir, base_id + '.jpg')
    img_dst = os.path.join(img_out_dir, base_id + '.jpg')
    shutil.copyfile(img_src, img_dst)


if not os.path.exists('data/processed/unlabeled/images'):
    os.mkdir('data/processed/unlabeled/images')

