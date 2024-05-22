# Clustering the Training Data Based on class distribution
# BoMeyering 2024

from sklearn.cluster import HDBSCAN
from sklearn.model_selection import train_test_split
import pandas as pd
from shutil import copyfile, copy2
from collections import Counter
import os
from tqdm import tqdm

# Read in pixel class data
filename = 'metadata/img_pixel_distributions.csv'
df = pd.read_csv(filename)
df = df.rename(columns = {'Unnamed: 0': 'number', 'keys': 'img_name'})
df['img_name'] = df['img_name'].apply(lambda x: x[2:-3])
df.drop(columns=['number'], inplace=True)

# Extract the keys from the data
keys = df['img_name']
df.drop(columns=['img_name'], inplace=True)

# Perform HDBSCAN clustering to generate separate distributions for the data to stratify over
hdb = HDBSCAN(min_cluster_size=100)
hdb.fit(df)

# Grab the predicted labels
labels = hdb.labels_

counts = Counter(labels)

split_df = pd.DataFrame(keys)
split_df['labels'] = labels
print(split_df)

train, test = train_test_split(split_df, train_size=.90, stratify=labels, )
train, val = train_test_split(train, train_size=.80, stratify=train['labels'])

print(train.size)
print(val.size)
print(test.size)

print(train['labels'].value_counts(normalize=True))
print(val['labels'].value_counts(normalize=True))
print(test['labels'].value_counts(normalize=True))

l_train_dir = './data/processed/labeled/train'
l_val_dir = './data/processed/labeled/val'
l_test_dir = './data/processed/labeled/test'

dirs = [l_train_dir, l_val_dir, l_test_dir]
for dir in dirs:
    if not os.path.exists(dir):
        os.mkdir(dir)
        os.mkdir(os.path.join(dir, 'images'))
        os.mkdir(os.path.join(dir, 'labels'))

# Move Train Images
train.to_csv('./metadata/train_img.csv')
val.to_csv('./metadata/val_img.csv')
test.to_csv('./metadata/test_img.csv')

for key in tqdm(train.img_name):
    base_key = key[:-4]
    # print(base_key)

    img_key = base_key+'.jpg'
    label_key = key

    old_img_path = os.path.join('./data/processed/labeled/images', img_key)
    img_dest = os.path.join(l_train_dir, 'images')
    copy2(old_img_path, img_dest)

    old_label_path = os.path.join('./data/processed/labeled/labels', label_key)
    label_dest = os.path.join(l_train_dir, 'labels')
    copy2(old_label_path, label_dest)

for key in tqdm(val.img_name):
    base_key = key[:-4]
    # print(base_key)

    img_key = base_key+'.jpg'
    label_key = key

    old_img_path = os.path.join('./data/processed/labeled/images', img_key)
    img_dest = os.path.join(l_val_dir, 'images')
    copy2(old_img_path, img_dest)

    old_label_path = os.path.join('./data/processed/labeled/labels', label_key)
    label_dest = os.path.join(l_val_dir, 'labels')
    copy2(old_label_path, label_dest)

for key in tqdm(test.img_name):
    base_key = key[:-4]
    # print(base_key)

    img_key = base_key+'.jpg'
    label_key = key

    old_img_path = os.path.join('./data/processed/labeled/images', img_key)
    img_dest = os.path.join(l_test_dir, 'images')
    copy2(old_img_path, img_dest)

    old_label_path = os.path.join('./data/processed/labeled/labels', label_key)
    label_dest = os.path.join(l_test_dir, 'labels')
    copy2(old_label_path, label_dest)