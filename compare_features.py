import argparse
import os
import pickle
import numpy as np

from datautil import sortFeatures

"""
The file used to compare the outlier features and the most close
inlier class features
"""

def parse_option():

    parser = argparse.ArgumentParser('argument for visulization')
    parser.add_argument("--inlier_features_path", type=str, default="/features/cifar10_resnet18_original_data__mixup_positive_alpha_0.1_beta_0.1_SimCLR_1.0_1.0_trail_0_128_256_500_train")
    parser.add_argument("--outlier_features_path", type=str, default="/features/cifar10_resnet18_original_data__mixup_positive_alpha_0.1_beta_0.1_SimCLR_1.0_1.0_trail_0_128_256_500_test_unknown")
    parser.add_argument("--num_classes", type=int, default=6)
    parser.add_argument("--target_class", type=int, default=5)     

    opt = parser.parse_args()

    opt.main_dir =os.getcwd()
    opt.similarities_save_path = opt.main_dir + opt.outlier_features_path + "_similarities"
    opt.inlier_features_path = opt.main_dir + opt.inlier_features_path
    opt.outlier_features_path = opt.main_dir + opt.outlier_features_path

    return opt


def similarities(target_features, outlier_features):

    similarities = []
    target_features = np.array(target_features)
    outlier_features = np.array(outlier_features)
    target_features = target_features.astype(np.double)
    outlier_features = outlier_features.astype(np.double)

    for outlier_feature in outlier_features:
        
        similarity = np.matmul(target_features, outlier_feature) / np.linalg.norm(target_features, axis=1) / np.linalg.norm(outlier_feature)
        similarities.append(np.max(similarity))

    return np.array(similarities)


if __name__ == "__main__":

    opt = parse_option()

    with open(opt.inlier_features_path, "rb") as f:
        features_inliers, _, _, labels_inliers = pickle.load(f)                       
        features_inliers = np.squeeze(np.array(features_inliers))

    with open(opt.outlier_features_path, "rb") as f:
        features_outliers, _, _, labels_outliers = pickle.load(f)                       
        features_outliers = np.squeeze(np.array(features_outliers))

    sorted_features = sortFeatures(features_inliers, labels_inliers, opt.num_classes)
    target_class_features = sorted_features[opt.target_class]

    #sorted_features_test = sortFeatures(features_outliers, labels_outliers, opt.num_classes)   #
    #features_outliers = sorted_features_test[opt.target_class]                                 #

    similarities = similarities(target_class_features, features_outliers)

    with open(opt.similarities_save_path, "wb") as f:
        pickle.dump(similarities, f)
    