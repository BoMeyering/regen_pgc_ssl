import torch

from s3torchconnector import S3MapDataset, S3IterableDataset, S3Data

# You need to update <BUCKET> and <PREFIX>
DATASET_URI="s3://kura-clover-datasets/K1702-kura-phenotyping/images/"
REGION = "us-east-1"

iterable_dataset = S3IterableDataset.from_prefix(DATASET_URI, region=REGION)

for item in iterable_dataset:
  print(item.key)

  