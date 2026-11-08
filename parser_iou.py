#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 20 20:00:11 2025

@author: zhi
"""


def iou_self(bbox1, bbox2):
    
    x1_min, y1_min, x1_max, y1_max = bbox1
    x2_min, y2_min, x2_max, y2_max = bbox2

    ix_min = max(x1_min, x2_min)
    iy_min = max(y1_min, y2_min)
    ix_max = min(x1_max, x2_max)
    iy_max = min(y1_max, y2_max)
    iw = max(0.0, ix_max - ix_min)
    ih = max(0.0, iy_max - iy_min)
    inter = iw * ih

    a1 = max(0.0, x1_max - x1_min) * max(0.0, y1_max - y1_min)
   
    return float(inter / (a1 + 1e-20))



import pickle


iou_file_path = "/home/zhi/projects/comprehensive_OSR_copy/save/SupCon/imagenet100_m_resnet18_mixup_positive_alpha_1.0_beta_1.0_layersaliencymix_4_SimCLR_1.0_0.5_0.05_lam0_trail_0_128_256_old_augmented_center/iou_15"

with open(iou_file_path, "rb") as f:
    all_ious = pickle.load(f)
    
    
mean_ious = []
num_epochs = len(all_ious)


for e in range(num_epochs):
    ious_epochs = all_ious[e]
    mean_ious_epochs = []
    for ious_epoch in ious_epochs:
        ious_iters = []
        for iou_iter in ious_epoch: 
            iou, annotation, mask = iou_iter
            ious_iters.append(iou)                          # iou_self(mask, annotation.tolist())
        mean_ious_epochs.append(sum(ious_iters) / len(ious_iters))
    
    mean_ious.append(sum(mean_ious_epochs) / len(mean_ious_epochs))
    
print(mean_ious)
    


import matplotlib.pyplot as plt

plt.plot(list(range(num_epochs)), mean_ious)
plt.xlabel("Epochs")
plt.ylabel("IoU")


    
    