import torch
import io
from PIL import Image
from s3torchconnector import S3MapDataset, S3IterableDataset

# You need to update <BUCKET> and <PREFIX>
DATASET_URI="s3://kura-clover-datasets/K1702-kura-phenotyping/images/"
REGION = "us-east-1"

image_dataset = S3MapDataset.from_prefix(DATASET_URI, region=REGION)
print(len(image_dataset))
for i in range(len(image_dataset)):
    # print(image_dataset[i])
    object = image_dataset[i].read()

    img_data = io.BytesIO(object)
    # content = object.read()
    img = Image.open(img_data)
    print(type(img))

# iterable_dataset = S3IterableDataset.from_prefix(DATASET_URI, region=REGION)

# for item in iterable_dataset:
#   print(item.key)

