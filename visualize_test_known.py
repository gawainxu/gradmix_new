import torch

import os
import cv2
import argparse
import pickle
import numpy as np
from scipy.spatial.distance import mahalanobis
import matplotlib.pyplot as plt

from distance_utils  import sortFeatures
from datautil import get_test_datasets, get_train_datasets
from datautil import osr_splits_inliers, osr_splits_outliers

"""
The file to visualize the misclassified test known samples
"""


def parse_option():

    parser = argparse.ArgumentParser('argument for visualization outliers')
    parser.add_argument("--train_feature_path", type=str, default="/features/cifar10_resnet18_original_data__mixup_positive_alpha_0.1_beta_0.1_SimCLR_1.0_1.0_trail_0_128_256_500_train")
    parser.add_argument("--test_known_feature_path", type=str, default="/features/cifar10_resnet18_original_data__mixup_positive_alpha_0.1_beta_0.1_SimCLR_1.0_1.0_trail_0_128_256_500_test_known")
    parser.add_argument("--num_classes", type=int, default=6)
    parser.add_argument("--datasets", type=str, default="cifar10")
    parser.add_argument("--trail", type=int, default=0)
    parser.add_argument("--k_out", type=int, default=30)
    parser.add_argument("--action", type=str, default="testing_known")
    parser.add_argument("--img_save_path", type=str, default="/temp_cifar10_resnet18_original_data__mixup_positive_alpha_0.1_beta_0.1_SimCLR_0.0_1.0_trail_0_test_unknown/")
    parser.add_argument("--if_inlier", type=bool, default=False)
    parser.add_argument("--usage", type=str, default="statistic", choices=["save_img", "statistic"])
    parser.add_argument("--stat_save_path", type=str, default="/stats/")

    opt = parser.parse_args()
    opt.main_dir = os.getcwd()
    opt.train_feature_path = opt.main_dir + opt.train_feature_path
    opt.test_known_feature_path = opt.main_dir + opt.test_known_feature_path
    opt.img_save_path = opt.main_dir + opt.img_save_path
    opt.stat_save_path = opt.main_dir + opt.stat_save_path + opt.test_known_feature_path.split("/")[-1]

    return opt


def pick_images(dataset, preds, labels, distances, opt, train_dataset=None, closest_indices=None):

    """
    ATTENTION: the real label is not realiable here, since the test_known features are downsampled
    """

    images = []
    closests = []
    for i, (img, label_data) in enumerate(dataset):

        #if i > 3000:
        #    break
        
        pred = preds[i]
        label = labels[i]
        print(i, label_data, label)

        if pred == label:
            continue

        print(label, pred)

        dis = distances[i]
        img = img.numpy()

        if img.shape[0] == 3:
            img = np.transpose(img, (1,2,0))
      
        save_path = opt.img_save_path + str(label) + "_" + str(pred) + "_" + str(round(dis*100)) + "_"  + str(i) + ".png"

        if train_dataset is not None:
            cloest_train_idx = closest_indices[i]
            cloest_train_img, _ = train_dataset[cloest_train_idx]
            save_path_clost_train = opt.img_save_path + str(label) + "_" + str(pred) + "_" + str(round(dis*100)) + "_" + str(i) + "_train.png"
            
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


def statistic(labels, preds, opt):

    # create a dictionary to record the results
    stat_dict = {}
    for c1 in range(opt.num_classes):
        stat_dict[str(c1)] = {}
        for c2 in range(opt.num_classes):
            stat_dict[str(c1)][str(c2)] = 0
    
    for l, p in zip(labels, preds):
        stat_dict[str(l)][str(p)] += 1

    return stat_dict


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
    similarity = np.squeeze(similarity)
    closest_idx = np.argmax(similarity)
    similarity = np.max(similarity)

    return similarity, closest_idx
    


if __name__ == "__main__":
    
    opt = parse_option()

    opt.inlier_classes = osr_splits_inliers[opt.datasets][opt.trail]
    opt.ourlier_classes = osr_splits_outliers[opt.datasets][opt.trail]

    with open(opt.train_feature_path, "rb") as f:
        features_train, _, _, labels_train = pickle.load(f)   

    with open(opt.test_known_feature_path, "rb") as f:
        features_known, _, _, labels_known = pickle.load(f)
        print(labels_known)

    sorted_features_train = sortFeatures(features_train, labels_train, opt)

    distances = []
    preds = []
    closest_indices = []
    labels = []
    for i, (feature, label) in enumerate(zip(features_known, labels_known)):
        diss = []
        cis = []
        for nc in range(opt.num_classes):
            #real_class_num = opt.inlier_classes[nc]
            dis, ci = similarities(sorted_features_train[nc], feature)
            diss.append(dis)
            cis.append(nc*5000 + ci)                       # between different classes

        pred = np.argmax(np.array(diss))
        closest_indices.append(cis[pred])
        #pred = opt.inlier_classes[pred]                   # no need to use real class number
        #print(i, label, pred)
        preds.append(pred)
        labels.append(label)
        distances.append(np.max(np.array(diss)))

    if opt.usage == "save_img":
        train_dataset = get_train_datasets(opt)
        print("train_dataset", len(train_dataset))
        visualize_dataset = get_test_datasets(opt)
        print("visualize_dataset", len(visualize_dataset))
        images = pick_images(dataset=visualize_dataset, preds=preds, labels=labels, distances=distances, opt=opt, train_dataset=train_dataset, closest_indices=closest_indices)
    else:
        stat_dict = statistic(labels=labels, preds=preds, opt=opt)
        print(stat_dict)
        with open(opt.stat_save_path, "wb") as f:
            pickle.dump(stat_dict, f)
