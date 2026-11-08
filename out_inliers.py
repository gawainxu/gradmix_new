import torch

import os
import cv2
import argparse
import pickle
import numpy as np
from scipy.spatial.distance import mahalanobis
import matplotlib.pyplot as plt

from util import  feature_stats
from distance_utils  import sortFeatures, sortData
from datautil import get_train_datasets, get_outlier_datasets
from datautil import osr_splits_inliers, osr_splits_outliers
from feature_reading import breaks

"""
The file to visualize the inlier samples that are far from the center and close to the outliers
"""


def parse_option():

    parser = argparse.ArgumentParser('argument for visualization outliers')
    parser.add_argument("--train_feature_path", type=str, default="/features/cifar10_resnet18_original_data__mixup_positive_alpha_0.1_beta_0.1_SimCLR_1.0_1.2_trail_0_128_256_500_train")
    parser.add_argument("--outlier_feature_path", type=str, default="/features/cifar10_resnet18_original_data__mixup_positive_alpha_0.1_beta_0.1_SimCLR_1.0_1.2_trail_0_128_256_500_test_unknown")
    parser.add_argument("--num_classes", type=int, default=6)
    parser.add_argument("--datasets", type=str, default="cifar10")
    parser.add_argument("--trail", type=int, default=0)
    parser.add_argument("--k_out", type=int, default=30)
    parser.add_argument("--action", type=str, default="outlier_visualization")
    parser.add_argument("--img_save_path", type=str, default="/temp_cifar10_resnet18_mixup_positive_alpha_0.1_beta_0.1_SimCLR_1.0_1.2_trail_0_test_unknown/")

    opt = parser.parse_args()
    opt.main_dir = os.getcwd()
    opt.train_feature_path = opt.main_dir + opt.train_feature_path
    opt.outlier_feature_path = opt.main_dir + opt.outlier_feature_path
    opt.img_save_path = opt.main_dir + opt.img_save_path

    return opt


def show_img(img, class_id, id, opt):

    img = img.numpy()
    if img.shape[0] == 3:
        img = np.transpose(img, (1,2,0))
        
    save_path = opt.img_save_path + "_" + str(class_id) + "_" + str(id) + "_" + ".png"
    print(save_path)
            
    if opt.datasets == "mnist":
        img = np.transpose(img, (1, 2, 0))
        print(img.shape)
        cv2.imwrite(save_path, img*255, [cv2.IMWRITE_PNG_COMPRESSION, 0])
    else:
        img = (1/(2 * 3)) * img + 0.5
        plt.imsave(save_path, img)
        
    return None



if __name__ == "__main__":
    
    opt = parse_option()

    opt.inlier_classes = osr_splits_inliers[opt.datasets][opt.trail]
    opt.ourlier_classes = osr_splits_outliers[opt.datasets][opt.trail]

    with open(opt.train_feature_path, "rb") as f:
        features_train, _, _, labels_train = pickle.load(f)   

    with open(opt.outlier_feature_path, "rb") as f:
        features_outlier, _, _, _ = pickle.load(f)

    sorted_features_train = sortFeatures(features_train, labels_train, opt)
    stats = feature_stats(sorted_features_train)
    dataset = get_train_datasets(opt)
    sorted_data = sortData(dataset, opt)
    #imgs = downsample_dataset(dataset, 500, 300, opt.num_classes)
    #print("downsampled dataset ", len(imgs))

    distances = []
    preds = []
    for c in range(opt.num_classes):
        features_c = sorted_features_train[c]
        data_c = sorted_data[c]
        for i, feature_c in enumerate(features_c):
            all_anchors = np.concatenate((features_c, features_outlier), axis=0)     # all the inliers of the same class and outliers
            binary_label_ones = np.array([1 for i in range(len(features_c))])
            binary_label_zeros = np.array([0 for i in range(len(features_outlier))])
            binary_label = np.concatenate((binary_label_ones, binary_label_zeros), axis=0)
            all_anchors = all_anchors.astype(np.double)
            similarities = np.matmul(all_anchors, feature_c) / np.linalg.norm(all_anchors, axis=1) / np.linalg.norm(feature_c)
            ind = np.argsort(similarities)[-10:]
            labels = binary_label[ind]
            if np.sum(labels) < len(labels) - 1:                                # if there is more than 1 zero in labels
                show_img(img=data_c[i], class_id=c, id=i, opt=opt) 