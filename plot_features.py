import matplotlib.pyplot as plt
import pickle
import numpy as np
import argparse
import os

from distance_utils  import sortFeatures


def parse_option():

    parser = argparse.ArgumentParser('argument for visulization')
    parser.add_argument("--feature_map_path", type=str, default="/features/cifar10_resnet18_temp_0.01_id_0_lr_0.001_bz_256_train")
    parser.add_argument("--num_classes", type=int, default=6)
    parser.add_argument("--save_path", type=str, default="/plots/cifar10_resnet18_temp_0.01_id_0_lr_0.001_bz_256_train.pdf")

    opt = parser.parse_args()
    opt.main_dir =os.getcwd()
    opt.feature_map_path = opt.main_dir + opt.feature_map_path
    opt.save_path = opt.main_dir + opt.save_path

    return opt


if __name__ == "__main__":

    opt = parse_option()

    with open(opt.feature_map_path, "rb") as f:
            feature_map, _, labels = pickle.load(f)

    sorted_features = sortFeatures(feature_map, labels, opt)
    
    fig, axs = plt.subplots(1, opt.num_classes, figsize=(12, 5.5), constrained_layout=True)
    fig.suptitle("Feature", fontsize=14)
    #plt.subplots_adjust(wspace=0, hspace=0.1)

    images = []
    for i in range(opt.num_classes):
        
        feature_map = sorted_features[i]
        feature_map = np.array(feature_map[:500]).astype(np.float32)                                
        images.append(axs[i].imshow(feature_map, cmap="Blues"))    
        axs[i].set_title("Class" + str(i), fontsize=8)
        #axs[1, i].set_xlabel('Feature Indice', fontsize=8)
        #axs[1, i].set_ylabel('Sample Indice', fontsize=8)
        axs[i].label_outer()

    fig.colorbar(images[5], ax=axs, orientation='horizontal', fraction=.1)
    fig.savefig(opt.save_path)
