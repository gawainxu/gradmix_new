#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 20 16:00:10 2021

@author: zhi
"""

import json
import pickle
import numpy as np


def featureMerge(featureList, opt):
    
    featureMaps = []
    labels = []
    print(opt.save_path_all)

    for featurePath in featureList:
        
        with open(featurePath, "rb") as f:
            features, labels_part = pickle.load(f)
  
        featureMaps = featureMaps + features
        labels = labels + labels_part
        
    featureMaps = np.array(featureMaps, dtype=object)
    #print("featureMaps_backbone", featureMaps_backbone.shape)
    featureMaps = np.squeeze(featureMaps)
    labels = np.squeeze(np.array(labels))
    
    with open(opt.save_path_all, 'wb') as f:
        pickle.dump((featureMaps, labels), f)
        
