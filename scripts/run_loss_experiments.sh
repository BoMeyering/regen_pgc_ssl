#!/bin/bash

python train.py configs/train_config_ce_loss.yaml
python train.py configs/train_config_ce_weighted_loss.yaml
python train.py configs/train_config_focal_loss.yaml
python train.py configs/train_config_cb_ce_loss.yaml
python train.py configs/train_config_cb_focal_loss.yaml
python train.py configs/train_config_acb_ce_loss.yaml
python train.py configs/train_config_acb_focal_loss.yaml
python train.py configs/train_config_recall_loss.yaml
python train.py configs/train_config_dice_loss.yaml
python train.py configs/train_config_tversky_loss.yaml

exit 0