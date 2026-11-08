import torch

import pickle
import numpy as np


def find_dominant(all_features, reduced_len):

    """
    find the dominant elements in class features
    """

    features_sum = np.sum(all_features, axis=0)
    removed_idx = np.argsort(np.abs(features_sum))[:reduced_len]

    all_features = np.delete(all_features, removed_idx, axis=1)
    return all_features, removed_idx