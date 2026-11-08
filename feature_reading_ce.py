#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 14 15:29:30 2022

@author: zhi
"""

import os
import platform
import sys
BASE_PATH = "/home/sysgen/Jiawen/causal_OSR"
sys.path.append(BASE_PATH) 

import argparse

import torch
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import numpy as np
import pickle
from itertools import chain

from networks.resnet_big import SupCEResNet
from networks.resnet_preact import SupConpPreactResNet
from networks.simCNN import simCNN_contrastive
from networks.mlp import SupConMLP
from networks.maskcon import MaskCon
from featureMerge_ce import featureMerge
from datautil import num_inlier_classes_mapping

from torch.utils.data import DataLoader
from datautil import get_train_datasets, get_test_datasets, get_outlier_datasets, osr_splits_inliers, osr_splits_outliers

torch.multiprocessing.set_sharing_strategy('file_system')


breaks = {"cifar-10-100-10": {"train": 5000, "test_known":500, "test_unknown": 50, "full": 100000}, 
          "cifar-10-100-50": {"train": 5000, "test_known": 500, "test_unknown": 50, "full": 100000}, 
           'cifar10':{"train": 5000, "test_known": 1100, "test_unknown": 500, "full": 100000}, 
           "tinyimgnet":{"train": 5000, "test_known": 50, "test_unknown": 50, "full": 100000}, 
           'mnist':{"train": 5000, "test_known": 500, "test_unknown": 500, "full": 100000}, 
           "svhn":{"train": 5000, "test_known": 500, "test_unknown": 500, "full": 100000}}

def parse_option():

    parser = argparse.ArgumentParser('argument for feature reading')

    parser.add_argument('--datasets', type=str, default='cifar10',
                        choices=["cifar-10-100-10", "cifar-10-100-50", 'cifar10', "tinyimgnet", 'mnist', "svhn"], help='dataset')
    parser.add_argument('--data_folder', type=str, default=None, help='path to custom dataset')
    parser.add_argument('--model', type=str, default="resnet18", choices=["resnet18", "resnet34", "resnet50", "preactresnet18", "preactresnet34", "simCNN", "MLP", "MaskCon"])
    parser.add_argument("--resnet_wide", type=int, default=1, help="factor for expanding channels in wide resnet")
    parser.add_argument("--model_path", type=str, default="/save/CE/cifar10_models/cifar10_resnet18_1trail_0_128_256_randaug2_6/last.pth")
    parser.add_argument("--trail", type=int, default=0)
    parser.add_argument("--split_train_val", type=bool, default=True)
    parser.add_argument("--action", type=str, default="feature_reading",
                        choices=["training_supcon", "trainging_linear", "testing_known", "testing_unknown", "feature_reading"])
    parser.add_argument('--method', type=str, default='SupCon',
                        choices=['SupCon', 'SimCLR', "MaskCon"], help='choose method')
    parser.add_argument("--feature_save", type=str, default="/features/")

    parser.add_argument("--epoch", type=int, default = 100)
    parser.add_argument("--feat_dim", type=int, default=128)

    parser.add_argument("--lr", type=str, default=0.01)
    parser.add_argument("--training_bz", type=int, default=400)
    parser.add_argument("--if_train", type=str, default="test_unknown", choices=['train', 'val', 'test_known', 'test_unknown', "full"])
    parser.add_argument('--batch_size', type=int, default=1, help='batch_size')
    parser.add_argument('--num_workers', type=int, default=4, help='num of workers to use')


    opt = parser.parse_args()

    opt.main_dir = os.getcwd()
    opt.model_path = opt.main_dir + opt.model_path
    opt.feature_save = opt.main_dir + opt.feature_save

    opt.n_cls = len(osr_splits_inliers[opt.datasets][opt.trail])
    opt.n_outs = len(osr_splits_outliers[opt.datasets][opt.trail])

    opt.break_idx = breaks[opt.datasets][opt.if_train]
    if platform.system() == 'Windows':
        opt.model_name = opt.model_path.split("\\")[-2]
    elif platform.system() == 'Linux':
        opt.model_name = opt.model_path.split("/")[-2]
    opt.save_path_all = opt.feature_save + opt.model_name + "_" + str(opt.epoch) + "_" + opt.if_train

    opt.num_classes = num_inlier_classes_mapping[opt.datasets]

    return opt



def set_model(opt):

    model = SupCEResNet(name=opt.model, num_classes=opt.num_classes)
    model = load_model(model, opt.model_path)
    model.eval()
    model = model.cpu()

    return model


def load_model(model, path):
    ckpt = torch.load(path, map_location='cpu')
    state_dict = ckpt['model']

    new_state_dict = {}
    for k, v in state_dict.items():
        k = k.replace("module.", "")
        new_state_dict[k] = v
    state_dict = new_state_dict
    model.load_state_dict(state_dict)
    
    return model



def normalFeatureReading(data_loader, model, opt):
    
    outputs = []
    labels = []

    for i, (img, label) in enumerate(data_loader):
        
        print(i)
        if i > opt.break_idx:
            break

        output_encoder = model.encoder(img)
        output_encoder = torch.squeeze(output_encoder)          
              
        outputs.append(output_encoder.detach().numpy())
        labels.append(label.numpy())

    with open(opt.save_path, "wb") as f:
        pickle.dump((outputs, labels), f)
        
            
def meanList(l):
    
    if len(l) == 0:
        return 0
    else:
        return sum(l)*1.0 / len(l)


def set_data(opt, class_idx=None):

    if opt.if_train == "train" or opt.if_train == "full":
        datasets = get_train_datasets(opt, class_idx)
    elif opt.if_train == "test_known":
        datasets = get_test_datasets(opt, class_idx)
    elif opt.if_train == "test_unknown":
        datasets = get_outlier_datasets(opt)

    return datasets
        

if __name__ == "__main__":
    
    opt = parse_option()

    model = set_model(opt)
    model = load_model(model, opt.model_path)
    print("Model loaded!!")
    
    featurePaths= []

    if opt.if_train == "train" or opt.if_train == "test_known" or opt.if_train == "full":
        for r in range(0, opt.n_cls):                 
            opt.save_path = opt.feature_save + "/temp" + str(r)
            featurePaths.append(opt.save_path)
            datasets = set_data(opt, class_idx=r)
            dataloader = DataLoader(datasets, batch_size=1, shuffle=False, sampler=None, 
                                    num_workers=1)
            normalFeatureReading(dataloader, model, opt)

        featureMerge(featurePaths, opt)

    else:
         for r in range(0, opt.n_outs):                            
            opt.save_path = opt.feature_save + "/temp" + str(r)
            featurePaths.append(opt.save_path)
            datasets = set_data(opt, class_idx=r)
            dataloader = DataLoader(datasets, batch_size=1, shuffle=False, sampler=None, 
                                    num_workers=1)
            normalFeatureReading(dataloader, model, opt)

         featureMerge(featurePaths, opt)
