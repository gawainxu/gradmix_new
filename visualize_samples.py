import torch

import os
import argparse
import pickle
import numpy as np
from scipy.spatial.distance import mahalanobis
import matplotlib.pyplot as plt
import cv2

from util import  feature_stats
from distance_utils  import sortFeatures
from datautil import get_train_datasets
from datautil import osr_splits_inliers

"""
The file to visualize the samples that are far from the center and close to the center
"""


def parse_option():

    parser = argparse.ArgumentParser('argument for visualization outliers')
    parser.add_argument("--train_feature_path", type=str, default="/features/cifar10_resnet18_original_data__vanilia__SimCLR_0.0_trail_5_128_400_train")
    parser.add_argument("--num_classes", type=int, default=6)
    parser.add_argument("--datasets", type=str, default="cifar10")
    parser.add_argument("--trail", type=int, default=5)
    parser.add_argument("--k_out", type=int, default=30)
    parser.add_argument("--k_in", type=int, default=50)
    parser.add_argument("--action", type=str, default="outlier_visualization")
    parser.add_argument("--img_save_path", type=str, default="/temp_cifar10/")

    opt = parser.parse_args()
    opt.main_dir = os.getcwd()
    opt.train_feature_path = opt.main_dir + opt.train_feature_path
    opt.img_save_path = opt.main_dir + opt.img_save_path

    return opt


def pick_images(dataset, index, class_id, group, opt, features_c=None, stats=None, nc=0):

    images = []
    for i, ind in enumerate(index):
        img, _ = dataset[ind]
        img = img.numpy()
        if img.shape[0] == 3:
            img = np.transpose(img, (1,2,0))
            
        if features_c is not None:
            feature = features_c[ind]
            diss = []
            for c in range(opt.num_classes):
                mu, var = stats[c]
                dis = mahalanobis(feature, mu, np.linalg.inv(var))
                diss.append(dis)
            pred = np.argmin(np.array(diss))
            pred = opt.inlier_classes[pred]
            dis = diss[nc]
            save_path = opt.img_save_path + str(class_id) + "_" + str(pred) + "_" + group + "_" + str(i) + "_" + str(ind) + "_" + str(int(dis)) + ".png"
        else:
            save_path = opt.img_save_path + str(class_id) + "_" + group + "_" + str(i) + "_" + str(ind) + ".png"
            
        if opt.datasets == "mnist":
            img = np.transpose(img, (1, 2, 0))
            cv2.imwrite(save_path, img*255, [cv2.IMWRITE_PNG_COMPRESSION, 0])
        else:
            img = (1/(2 * 3)) * img + 0.5
            plt.imsave(save_path, img)
        images.append(images)
        
    return images


if __name__ == "__main__":
    
    opt = parse_option()

    opt.inlier_classes = osr_splits_inliers[opt.datasets][opt.trail]

    with open(opt.train_feature_path, "rb") as f:
        features_train, _, _, labels_train = pickle.load(f)   

    sorted_features_train = sortFeatures(features_train, labels_train, opt)
    stats = feature_stats(sorted_features_train)

    distances = []
    for nc in range(opt.num_classes):
        real_class_num = opt.inlier_classes[nc]
        mu, var = stats[nc]
        features_train_c = sorted_features_train[nc]
        dis_c = []
        for features in features_train_c:
            dis = mahalanobis(features, mu, np.linalg.inv(var))     # np.matmul(features, mu)  #
            dis_c.append(dis)

        distances.append(dis_c)
        dataset = get_train_datasets(opt, class_idx=nc)
        ascending_order = np.argsort(dis_c)

        # pick the fartest ones
        index = ascending_order[-opt.k_out:]                    # the last one is the most unsimilar one
        images = pick_images(dataset=dataset, index=index, class_id=real_class_num, group="out", 
                             opt=opt, features_c=features_train_c, stats=stats, nc=nc)

        # pick the closest ones
        index = ascending_order[:opt.k_in]                    # the first one is the most similar one
        images = pick_images(dataset=dataset, index=index, class_id=real_class_num, group="in", opt=opt, features_c=features_train_c, stats=stats, nc=nc)      