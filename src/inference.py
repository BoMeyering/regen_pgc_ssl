# Inference Script
# BoMeyering 2024

import torch
import cv2
import numpy as np
import random



def show_img(img):
    cv2.namedWindow('test', cv2.WINDOW_NORMAL)
    cv2.imshow('test', img)
    cv2.waitKey()
    cv2.destroyAllWindows()

def map_preds(preds, mapping):
    """preds = integer mask of shape (H, W)"""

    h, w = preds.shape
    color_mask = np.zeros(shape=(h, w, 3)).astype(np.uint8)
    for i in np.unique(preds):
        idx = np.where(preds==i)
        rgb = mapping.get(i)
        color_mask[idx] = np.array(rgb)
    
    return color_mask

def overlay_preds(img, color_mask, alpha, gamma=0.0):
    beta = 1-alpha
    overlay = cv2.addWeighted(img, alpha, color_mask, beta, gamma)
    return overlay


    

    return color_mask


if __name__ == '__main__':

    from skimage.transform import resize
    img = cv2.imread('data/raw/cropandweed-dataset/data/images/ave-0007-0000.jpg')

    # Read in mask and resize to model output size
    preds = cv2.imread('data/raw/cropandweed-dataset/data/labelIds/CropAndWeed/ave-0007-0000.png', cv2.IMREAD_GRAYSCALE)
    mod_out = resize(preds, (1000, 1000), order=0)

    # Now resize to original image size
    preds = resize(mod_out, img.shape[:2], order=0)
    # print(img.shape[:2:-1])
    # print(preds.shape)

    show_img(img)
    show_img(preds)
    mapping = {}
    num_classes = 5
    for i in np.unique(preds):
        mapping[i] = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

    color_mask = map_preds(preds, mapping=mapping)
    show_img(color_mask)

    overlay = overlay_preds(img, color_mask, 0.5)
    show_img(overlay)

    # over2 = cv2.resize(overlay, (512, 512))
    # show_img(over2)

    # print(preds.shape)
    # preds2 = np.expand_dims(preds, axis = 2)
    # preds2 = np.repeat(preds2, 3, 2)

    # print(preds)
    # print(preds2)

    # idx = np.where(preds == 3)
    # print(preds2[idx])