# Inference Testing
# BoMeyering 2024

import torch
import cv2
import argparse
import numpy as np
import torch.nn.functional as F
from random import randint
import os
import matplotlib.pyplot as plt

from src.models import create_smp_model
from src.inference import show_img, map_preds, overlay_preds
from src.transforms import get_val_transforms

image_paths = [
    'data/processed/unlabeled/images/0cc68341-d7ee-4135-a8d8-d7ae3cfce7a0.jpg',
    'data/processed/unlabeled/images/00a5ce87-6d28-448d-bf85-60a385d5eb63.jpg',
    'data/processed/labeled/train/images/vwg-1361-0007.jpg',
    'data/processed/labeled/train/images/D__aeef9ae6770eaf89dd5a69af7b2f7cff.jpg',
    'data/processed/unlabeled/images/0a4f1984-5a04-4430-8480-0b281a6bdcac.jpg',
    'data/processed/unlabeled/images/0a73b997-f052-4a64-aa4c-0c2d1dc28369.jpg'
    ]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

args = argparse.Namespace()
setattr(args, 'model', argparse.Namespace)
model_args = {
    'model_name': 'deeplabv3plus',
    'num_classes': 8,
    'in_channels': 3,
    'encoder_name': 'efficientnet-b4',
    'encoder_weights': 'imagenet', 
    'encoder_depth': 5,
}
for k, v in model_args.items():
    setattr(args.model, k, v)

# print(vars(args.model))

mapping = {
    0: (0, 0, 0), # soil
    1: (255, 255, 255), # quadrat
    2: (255, 0, 0), # grass
    3: (0, 255, 0), # clover
    4: (0, 0, 255), # weeds
    5: (0, 75, 55), # corn
    6: (12, 142, 194), # soybean
    7: (133, 12, 194) # other_vegetation
}

model = create_smp_model(args)

state_dict = torch.load('model_checkpoints/large_1024_models/dlv3p_1024_enb4_recall_ce_2024-06-07_00.09.45_epoch_18_2024-06-08_04.28.23', map_location=device)['model_state_dict']
# state_dict = torch.load('model_checkpoints/large_1024_fixmatch_models/fixmatch_dlv3p_1024_enb4_ce_unweighted_2024-06-10_15.30.51_epoch_9_2024-06-12_22.08.45', map_location=device)['model_state_dict']
model.load_state_dict(state_dict)
model = model.to(device)
model.eval()

inf_tfms = get_val_transforms((1024, 1024))

with torch.no_grad():
    for path in image_paths:
        basename = os.path.basename(path)
        img = cv2.imread(path)
        # show_img(img)
        h, w = img.shape[:2]
        t_img = inf_tfms(image=img)['image'].unsqueeze(0).to(device)

        logits = model(t_img)

        sm = F.softmax(logits, dim=1)
        
        max_vals, y = torch.max(sm, dim=1)
        max_vals = max_vals.squeeze(0)
        # ge_idx = torch.ge(max_vals, .95)
        # max_vals = (max_vals - 0.85)/(1.00 - .15) * ge_idx
        # max_vals = max_vals

        ge_idx = torch.ge(max_vals, 0.95).cpu().numpy().astype('float')
        ge_idx = cv2.resize(ge_idx, (w, h)).astype(bool)
        max_vals = cv2.resize(max_vals.cpu().numpy(), (w, h))
        threshed = (max_vals - 0.85)/(100 - 0.15) * ge_idx
        show_img(threshed*255)

        # plt.imshow(max_vals, cmap='viridis')
        # plt.savefig('_'.join(['conf', basename]), bbox_inches='tight', dpi=400)
        # plt.close()
        # plt.imshow(threshed, cmap='viridis')
        # plt.savefig('_'.join(['fixmatch', basename]), bbox_inches='tight', dpi=400)
        # # plt.show()
        # plt.close()


        preds = torch.argmax(sm, dim=1).squeeze().cpu().numpy()

        # print(preds.shape)

        color_mask = map_preds(preds, mapping)
        color_mask = cv2.resize(color_mask, (w, h))
        
        # print(color_mask.shape)
        # print(img.shape)
        # show_img(color_mask)
        print(color_mask.shape)
        print(ge_idx.shape)
        # color_mask *= ge_idx[:, :, np.newaxis]
        overlay = overlay_preds(img, color_mask, alpha=.4)
        show_img(overlay)

        # cv2.imwrite(filename="_".join(['pred', basename]), img=overlay)
        # cv2.imwrite(filename="_".join(['orig', os.path.basename(path)]), img=img)

        
