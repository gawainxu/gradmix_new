import torch

import os
import cv2
import argparse
import pickle
import numpy as np
from scipy.spatial.distance import mahalanobis
import matplotlib.pyplot as plt

from util import  feature_stats
from distance_utils  import sortFeatures
from datautil import get_test_datasets, get_outlier_datasets, get_train_datasets
from datautil import osr_splits_inliers, osr_splits_outliers
from feature_reading import breaks

"""
The file to visualize the samples that are far from the center and close to the center
"""


def parse_option():

    parser = argparse.ArgumentParser('argument for visualization outliers')
    parser.add_argument("--train_feature_path", type=str, default="/features/cifar10_resnet18_original_data__vanilia__SimCLR_only_trail_0_128_256_randaug2_6_400_train_0")
    parser.add_argument("--outlier_feature_path", type=str, default="/features/cifar10_resnet18_original_data__vanilia__SimCLR_only_trail_0_128_256_randaug2_6_400_test_unknown_0")
    parser.add_argument("--num_classes", type=int, default=1)
    parser.add_argument("--datasets", type=str, default="cifar10")
    parser.add_argument("--trail", type=int, default=0)
    parser.add_argument("--k_out", type=int, default=30)
    parser.add_argument("--action", type=str, default="outlier_visualization")
    parser.add_argument("--img_save_path", type=str, default="/temp_SimCLR_only_trail_0_128_256_randaug2_6/")
    parser.add_argument("--if_inlier", type=bool, default=False)

    opt = parser.parse_args()
    opt.main_dir = os.getcwd()
    opt.train_feature_path = opt.main_dir + opt.train_feature_path
    opt.outlier_feature_path = opt.main_dir + opt.outlier_feature_path
    opt.img_save_path = opt.main_dir + opt.img_save_path

    return opt


def pick_images(dataset, preds, distances, group, opt, train_dataset=None, closest_indices=None):

    images = []
    closests = []
    for i, (img, _) in enumerate(dataset):
        if i > 500:
            break
        img = img.numpy()
        if img.shape[0] == 3:
            img = np.transpose(img, (1,2,0))
        
        pred = preds[i]
        dis = distances[i]

        if dis > 0.8:
             closests.append(i)
      
        save_path = opt.img_save_path + str(pred) + "_" + str(round(dis*100)) + "_" + group + "_" + str(i) + ".png"

        if train_dataset is not None:
            cloest_train_idx = closest_indices[i]
            cloest_train_img, _ = train_dataset[cloest_train_idx]
            save_path_clost_train = opt.img_save_path + str(pred) + "_" + str(round(dis*100)) + "_" + group + "_" + str(i) + "_train.png"
            
        if opt.datasets == "mnist":
            img = np.transpose(img, (1, 2, 0))
            cv2.imwrite(save_path, img*255, [cv2.IMWRITE_PNG_COMPRESSION, 0])
            if train_dataset is not None:
                cloest_train_img = np.transpose(cloest_train_img, (1, 2, 0))
                cv2.imwrite(save_path_clost_train, cloest_train_img*255, [cv2.IMWRITE_PNG_COMPRESSION, 0])
        else:
            img = (1/(2 * 3)) * img + 0.5
            plt.imsave(save_path, img)
            if train_dataset is not None:
                cloest_train_img = cloest_train_img.numpy()
                cloest_train_img = np.transpose(cloest_train_img, (1,2,0))
                cloest_train_img = (1/(2 * 3)) * cloest_train_img + 0.5
                plt.imsave(save_path_clost_train, cloest_train_img)
        images.append(images)

    print(closests)
        
    return images


def downsample_dataset(dataset, ori_class_size, new_class_size, num_classes):
    
    imgs = []
    for nc in range(num_classes):
        for i in range(new_class_size):
            img, _ = dataset[nc*ori_class_size+i]
            imgs.append(img)

    return imgs


def similarities(target_features, visualize_feature):

    target_features = np.array(target_features)
    visualize_feature = np.array(visualize_feature)
    target_features = target_features.astype(np.double)
    visualize_feature = visualize_feature.astype(np.double)

    similarity = np.matmul(target_features, visualize_feature) / np.linalg.norm(target_features, axis=1) / np.linalg.norm(visualize_feature)
    print("similarity shape", similarity.shape)
    similarity = np.squeeze(similarity)
    closest_idx = np.argmax(similarity)
    similarity = np.max(similarity)
    print("similarity", similarity, closest_idx)

    return similarity, closest_idx
    


if __name__ == "__main__":
    
    opt = parse_option()

    opt.inlier_classes = osr_splits_inliers[opt.datasets][opt.trail]
    opt.ourlier_classes = osr_splits_outliers[opt.datasets][opt.trail]

    with open(opt.train_feature_path, "rb") as f:
        features_train, _, _, labels_train = pickle.load(f)   

    with open(opt.outlier_feature_path, "rb") as f:
        features_outlier, _, _, _ = pickle.load(f)

    sorted_features_train = sortFeatures(features_train, labels_train, opt)
    train_dataset = get_train_datasets(opt)
    if opt.if_inlier:
        visualize_dataset = get_test_datasets(opt)
    else:
        visualize_dataset = get_outlier_datasets(opt)

    distances = []
    preds = []
    closest_indices = []
    for i, feature in enumerate(features_outlier):
        diss = []
        cis = []
        for nc in range(opt.num_classes):
            real_class_num = opt.inlier_classes[nc]
            dis, ci = similarities(sorted_features_train[nc], feature)
            diss.append(dis)
            cis.append(ci)

        pred = np.argmax(np.array(diss))
        closest_indices.append(cis[pred])
        pred = opt.inlier_classes[pred]
        preds.append(pred)
        distances.append(np.max(np.array(diss)))

    images = pick_images(dataset=visualize_dataset, preds=preds, distances=distances, group="outlier", opt=opt, train_dataset=train_dataset, closest_indices=closest_indices) 