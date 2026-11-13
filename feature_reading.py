#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 14 15:29:30 2022

@author: zhi
"""

import os
import platform
import sys
BASE_PATH = "/home/sysgen/Jiawen/gradmix_applications"
sys.path.append(BASE_PATH) 

import argparse
import torch
import pickle


from networks.resnet_big import SupConResNet, LinearClassifier
from networks.mlp import SupConMLP
from networks.ts_models import basic_CNN
from featureMerge import featureMerge
from datautil import num_inlier_classes_mapping

from torch.utils.data import DataLoader
from datautil import get_train_datasets, get_test_datasets, get_outlier_datasets, osr_splits_inliers, osr_splits_outliers

torch.multiprocessing.set_sharing_strategy('file_system')


def parse_option():

    parser = argparse.ArgumentParser('argument for feature reading')

    parser.add_argument('--datasets', type=str, default='paderborn')
    parser.add_argument('--data_folder', type=str, default=None, help='path to custom dataset')
    parser.add_argument('--model', type=str, default="basic_cnn")
    parser.add_argument("--model_path", type=str, default="")
    parser.add_argument("--linear_model_path", type=str, default=None)
    parser.add_argument("--trail", type=int, default=0)
    parser.add_argument("--action", type=str, default="feature_reading",
                        choices=["training_supcon", "trainging_linear", "testing_known", "testing_unknown", "feature_reading"])
    parser.add_argument('--method', type=str, default='SupCon',
                        choices=['SupCon', 'SimCLR', "MaskCon"], help='choose method')
    parser.add_argument("--feature_save", type=str, default="/features/")

    # temperature
    parser.add_argument('--temp', type=float, default=0.05, help='temperature for loss')
    parser.add_argument("--feat_dim", type=int, default=128)

    parser.add_argument("--lr", type=str, default=0.01)
    parser.add_argument("--if_train", type=str, default="test_known",
                        choices=['train', 'val', 'test_known', 'test_unknown', "full", "curruption"])
    parser.add_argument('--num_workers', type=int, default=4, help='num of workers to use')
    parser.add_argument("--last_model_path", type=str, default=None)

    # mixup parameters
    parser.add_argument("--mixup_positive", type=bool, default=False)
    parser.add_argument("--positive_method", type=str, default="no", choices=["min_similarity", "random", "prob_similarity", "no"])

    opt = parser.parse_args()

    opt.main_dir = os.getcwd()
    opt.model_path = opt.main_dir + opt.model_path
    opt.feature_save = opt.main_dir + opt.feature_save
    if opt.linear_model_path is not None:
        opt.linear_model_path = opt.main_dir + opt.linear_model_path

    opt.n_cls = len(osr_splits_inliers[opt.datasets][opt.trail])
    opt.n_outs = len(osr_splits_outliers[opt.datasets][opt.trail])                                 # !!!!!!!!!!!!!

    if platform.system() == 'Windows':
        opt.model_name = opt.model_path.split("\\")[-2]
    elif platform.system() == 'Linux':
        opt.model_name = opt.model_path.split("/")[-2]
    opt.save_path_all = opt.feature_save + opt.model_name + "_" + opt.if_train

    return opt


def load_model(opt):

    if opt.datasets == "mnist":
        in_channels = 1
    else:
        in_channels = 3

    if opt.model == "resnet18" or opt.model == "resnet34":
        model = SupConResNet(name=opt.model, feat_dim=opt.feat_dim, in_channels=in_channels)
    elif opt.model == "basic_cnn":
        model = basic_CNN(feature_dim=opt.feat_dim, in_channels=opt.num_channels)

    ckpt = torch.load(opt.model_path, map_location='cpu')
    state_dict = ckpt['model']

    new_state_dict = {}
    for k, v in state_dict.items():
        k = k.replace("module.", "")
        new_state_dict[k] = v
    state_dict = new_state_dict

    model = model.cpu()
    model.load_state_dict(state_dict)
    model.eval()

    if opt.linear_model_path is not None:

        linear_model = LinearClassifier(name=opt.model, num_classes=opt.n_cls)
        ckpt = torch.load(opt.linear_model_path, map_location='cpu')
        state_dict = ckpt['model']
        linear_model = linear_model.cpu()
        linear_model.load_state_dict(state_dict)
        linear_model.eval()

        return model, linear_model
    else:
        return model, None


def normalFeatureReading(data_loader, model, linear_model, opt):
    
    outputs_backbone = []
    outputs_linear = []
    labels = []

    for i, (img, label) in enumerate(data_loader):
        
        print(i)
        if opt.method == "SupCon" or opt.method == "SimCLR":
            output = model(img)
            output = torch.squeeze(output)

        if linear_model is not None:
            linear_output = linear_model(model.encoder(img))
            outputs_backbone.append(output.detach().numpy())
            outputs_linear.append(linear_output.detach().numpy())
        else:
            outputs_backbone.append(output.detach().numpy())

        labels.append(label.numpy())

    with open(opt.save_path, "wb") as f:
        pickle.dump((outputs_backbone, outputs_linear, labels), f)



def set_data(opt, class_idx=None):

    if opt.if_train == "train" or opt.if_train == "full":
        datasets = get_train_datasets(opt)
    elif opt.if_train == "test_known":
        datasets = get_test_datasets(opt)
    elif opt.if_train == "test_unknown":
        datasets = get_outlier_datasets(opt)

    return datasets
        

if __name__ == "__main__":

    opt = parse_option()

    model, linear_model = load_model(opt)
    print("Model loaded!!")

    featurePaths = []

    if opt.if_train == "train" or opt.if_train == "test_known" or opt.if_train == "full":
        for r in range(0, opt.n_cls):
            print(r)
            opt.save_path = opt.feature_save + "temp" + str(r)
            featurePaths.append(opt.save_path)
            datasets = set_data(opt, class_idx=r)

            dataloader = DataLoader(datasets, batch_size=1, shuffle=False, sampler=None,
                                    num_workers=0)
            normalFeatureReading(dataloader, model, linear_model, opt)

        featureMerge(featurePaths, opt.save_path_all)

    else:
        datasets = set_data(opt)
        dataloader = DataLoader(datasets, batch_size=1, shuffle=False, sampler=None,
                                num_workers=0)
        opt.save_path = opt.save_path_all
        normalFeatureReading(dataloader, model, linear_model, opt)

        
        
        
