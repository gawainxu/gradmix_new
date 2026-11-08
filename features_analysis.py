import pickle
import argparse
import numpy as np
from distance_utils  import sortFeatures
from ID import id_2nn, euclidean_distance
from datautil import sortFeatures as sortFeatures_id


def parse_option():

    parser = argparse.ArgumentParser('argument for feature analysis')
    
    parser.add_argument("--feature_path", type=str, default="./features/tinyimgnet_resnet18_mixup_positive_alpha_1.0_beta_1.0_layersaliencymix_3,4_SimCLR_1.0_1.0_0.05_trail_6_128_256_200_test_known")
    parser.add_argument("--num_classes", type=int, default=100)

    opt = parser.parse_args()

    return opt


def intra_class_distances(sorted_features):
    
    diss = []
    for features in sorted_features:
        features = np.array(features, dtype=float)
        features = features / np.linalg.norm(features, axis=1, keepdims=True)
        dis = np.matmul(features, features.T)
        dis = np.triu(dis)
        np.fill_diagonal(dis, 0)
        dis = np.mean(dis)
        diss.append(dis)
        
    return sum(diss) / len(diss)


def intrinstic_dimension(feature_maps, labels, opt):
    # ID and eigenvalue for feature diversity
    sorted_features = sortFeatures_id(feature_maps, labels, opt.num_classes)
    ids = []

    for feature_maps_c in sorted_features:
        feature_maps_c = np.array(feature_maps_c)
        distances = euclidean_distance(feature_maps_c)
        _, _, d, r,pval = id_2nn(distances)
        ids.append(d)
    
    print("id mean: ", sum(ids)/len(ids))
    


def main():
    
    opt = parse_option()
    
    with open(opt.feature_path, "rb") as f:
        features_head, features_backbone, _, labels = pickle.load(f) 
    
    sorted_features_head = sortFeatures(features_head, labels, opt)
    sorted_features_backbone = sortFeatures(features_backbone, labels, opt)
    
    print(opt.feature_path)
    print("intra class distance is", intra_class_distances(sorted_features_head))
    intrinstic_dimension(features_head, labels, opt)

if __name__ == '__main__':
    main()
