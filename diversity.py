import os
import argparse
import pickle
import numpy as np

from ID import euclidean_distance
from datautil import sortFeatures

"""
the file used to test the intra-class feature diversity
"""

"""
Hyperspherical Potential Energy
https://arxiv.org/pdf/2104.02290
"""

def hpe_diversity(features, s=0):

    features = np.divide(features, np.linalg.norm(features, axis=1, keepdims=True))
    ecu_dis = euclidean_distance(features)
    ecu_dis = ecu_dis[~np.eye(ecu_dis.shape[0], dtype=bool)].reshape(-1)
    ecu_dis = np.reciprocal(ecu_dis)
    print(ecu_dis)

    if s == 0:
        hd = np.sum(np.log(ecu_dis))
    else:
        ecu_dis = np.power(ecu_dis, s)
        hd = np.sum(ecu_dis)
        
    hd /= len(ecu_dis)

    return hd


if __name__ == "__main__":
    
    feature_path = "./features/cifar10_resnet18_1trail_0_128_256_100_test_known"
    num_classes = 6
    s = 0

    #with open(feature_path, "rb") as f:
    #    feature_maps, feature_maps_backbone, _, labels = pickle.load(f)
    with open(feature_path, "rb") as f:
        feature_maps, labels = pickle.load(f)
    npoints = feature_maps.shape[0]

    if len(feature_maps.shape) > 2:
        feature_maps = np.reshape(feature_maps, (npoints, -1))
    
    #if len(feature_maps_backbone.shape) > 2:
    #    feature_maps_backbone = np.reshape(feature_maps_backbone, (npoints, -1))

    sorted_features = sortFeatures(feature_maps, labels, num_classes)
    hds = []

    for feature_maps_c in sorted_features:

        feature_maps_c = np.array(feature_maps_c)
        distances = euclidean_distance(feature_maps_c)
        hd = hpe_diversity(distances, s)
        print(hd)
        hds.append(hd)

    print("mean: ", sum(hds)/len(hds))

