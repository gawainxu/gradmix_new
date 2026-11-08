import torch

import os
import pickle
import numpy as np
import argparse

from distance_utils import EuclideanDistance, EuclideanStat, sortFeatures
from manipulate_features import find_dominant


def arg_parse():

    parser = argparse.ArgumentParser('argument for statistics')
    parser.add_argument("--inlier_feature_path", type=str, default="/features/cifar10_resnet18_temp_0.01_id_0_lr_0.001_bz_512_train")
    parser.add_argument("--num_classes", type=int, default=6)
    parser.add_argument("--reduced_len", type=int, default=0)

    opt = parser.parse_args()
    opt.main_dir = os.getcwd()
    opt.inlier_feature_path = opt.main_dir + opt.inlier_feature_path

    return opt


def interClassSeperation(means):
    
    """
    It is to compute the distances 
    """
    distances = []
    for i in range(0, len(means)):
        for j in range(i+1, len(means)):
            distances.append(EuclideanDistance(means[i], means[j]))
            
    return np.min(np.array(distances))



def intraClassSeperation(sortedFeatures, classCenters):
    
    num_instances  = 0
    sum_distances = 0
    for features, center in zip(sortedFeatures, classCenters):
        num_instances += len(features)
        for feature in features:
            sum_distances += EuclideanDistance(feature, center)
            
    return sum_distances / num_instances



if __name__ == "__main__":
        
    opt = arg_parse()
    
    with open(opt.inlier_feature_path, "rb") as f:
        featuresInliers, _, labelsInliers = pickle.load(f)

    if opt.reduced_len > 0:
        featuresInliers, reduced_idx = find_dominant(featuresInliers, opt.reduced_len)
    
    sortedFeatures = sortFeatures(featuresInliers, labelsInliers, opt)
    means = EuclideanStat(sortedFeatures)    
    
    print("Intra Seperation: ", intraClassSeperation(sortedFeatures, means))
    print("Inter Seperation: ", interClassSeperation(means))
    print("Ratio: ", intraClassSeperation(sortedFeatures, means)/interClassSeperation(means))