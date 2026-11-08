#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 20 16:00:10 2021

@author: zhi
"""

import argparse
import pickle
import numpy as np


def parse_option():
    parser = argparse.ArgumentParser('argument for feature analysis')

    parser.add_argument("--features_folder", type=str,
                        default="./features/imagenet100_vgg16_original_data__vanilia__Joint_0.4_0.6_trail_0_128_256_split_128_train")
    parser.add_argument("--save_path_all", type=str,
                        default="./features/imagenet100_vgg16_original_data__vanilia__Joint_0.4_0.6_trail_0_128_256_split_128_train_all")

    opt = parser.parse_args()

    return opt

def featureMerge(featureList, save_path_all):
    
    featureMaps = []
    featureMaps_backbone = []
    featureMaps_linear = []
    labels = []
    print(save_path_all)

    for featurePath in featureList:

        print(featurePath)
        with open(featurePath, "rb") as f:
            features, feature_backbone, feature_linear, labels_part = pickle.load(f)
  
        if len(feature_linear) > 0:
            featureMaps_linear = featureMaps_linear + feature_linear

        featureMaps_backbone = featureMaps_backbone + feature_backbone[:2000]
        featureMaps = featureMaps + features[:2000]
        labels = labels + labels_part[:2000]
        
    featureMaps_backbone = np.array(featureMaps_backbone, dtype=object)
    featureMaps = np.array(featureMaps, dtype=object)
    #print("featureMaps_backbone", featureMaps_backbone.shape)
    featureMaps_backbone = np.squeeze(featureMaps_backbone)
    featureMaps = np.squeeze(featureMaps)
    featureMaps_linear = np.squeeze(np.array(featureMaps_linear))
    labels = np.squeeze(np.array(labels))
    
    with open(save_path_all, 'wb') as f:
        pickle.dump((featureMaps, featureMaps_backbone, featureMaps_linear, labels), f)


if __name__ == '__main__':

    import os

    opt = parse_option()
    features_list = ["temp" + str(i) for i in range(100)]
    feature_list = [os.path.join(opt.features_folder, fl) for fl in features_list]

    featureMerge(feature_list, opt.save_path_all)


        
