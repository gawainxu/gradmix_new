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
import numpy as np
import pickle

from networks.resnet_multi import SupConResNet_end, SupConResNet_inter
from datautil import num_inlier_classes_mapping

from torch.utils.data import DataLoader
from datautil import get_train_datasets, get_test_datasets, get_outlier_datasets, osr_splits_inliers, \
    osr_splits_outliers

torch.multiprocessing.set_sharing_strategy('file_system')

breaks = {"cifar-10-100-10": {"train": 5000, "test_known": 500, "test_unknown": 50, "full": 100000},
          "cifar-10-100-50": {"train": 5000, "test_known": 500, "test_unknown": 50, "full": 100000},
          'cifar10': {"train": 5000, "test_known": 500, "test_unknown": 500, "full": 100000},
          "tinyimgnet": {"train": 5000, "test_known": 100, "test_unknown": 20, "full": 100000},
          'mnist': {"train": 5000, "test_known": 500, "test_unknown": 500, "full": 100000},
          "svhn": {"train": 5000, "test_known": 500, "test_unknown": 500, "full": 100000},
          "cub": {"train": 5000, "test_known": 500, "test_unknown": 500, "full": 100000}}


def parse_option():
    parser = argparse.ArgumentParser('argument for feature reading')

    parser.add_argument('--datasets', type=str, default='cifar10',
                        choices=["cifar-10-100-10", "cifar-10-100-50", 'cifar10', "tinyimgnet", 'mnist', "svhn", "cub",
                                 "aircraft"], help='dataset')
    parser.add_argument('--data_folder', type=str, default=None, help='path to custom dataset')
    parser.add_argument('--model', type=str, default="resnet18",
                        choices=["resnet18", "resnet34", "preactresnet18", "preactresnet34"])
    parser.add_argument("--model_path", type=str,
                        default="/save/SupCon/cifar10_resnet18_ensemble_trail_1_128_256_end_0.5_0.05_0.005_1.0_1.0_1.0/last.pth")
    parser.add_argument("--trail", type=int, default=1)
    parser.add_argument("--split_train_val", type=bool, default=True)
    parser.add_argument("--action", type=str, default="feature_reading",
                        choices=["training_supcon", "trainging_linear", "testing_known", "testing_unknown",
                                 "feature_reading"])
    parser.add_argument('--method', type=str, default='SupCon',
                        choices=['SupCon', 'SimCLR'], help='choose method')
    parser.add_argument("--feature_save", type=str, default="/features/")

    # temperature
    parser.add_argument('--temp1', type=float, default=0.05, help='temperature for loss1')
    parser.add_argument('--temp2', type=float, default=0.05, help='temperature for loss2')
    parser.add_argument('--temp3', type=float, default=0.05, help='temperature for loss3')

    parser.add_argument('--alpha1', type=float, default=1, help='coefficient for loss1')
    parser.add_argument('--alpha2', type=float, default=1, help='coefficient for loss2')
    parser.add_argument('--alpha3', type=float, default=1, help='coefficient for loss3')

    parser.add_argument("--epoch", type=int, default=600)
    parser.add_argument("--augmentation_method", type=str, default="vanilia",
                        choices=["vanilia", "upsampling", "mixup"])
    parser.add_argument("--ensemble_mode", type=str, default="end")
    parser.add_argument("--ensemble_num", type=int, default=1)
    parser.add_argument("--feat_dim", type=int, default=128)

    parser.add_argument("--lr", type=str, default=0.01)
    parser.add_argument("--training_bz", type=int, default=600)
    parser.add_argument("--if_train", type=str, default="train",
                        choices=['train', 'val', 'test_known', 'test_unknown', "full"])
    parser.add_argument('--batch_size', type=int, default=1, help='batch_size')
    parser.add_argument('--num_workers', type=int, default=4, help='num of workers to use')

    # upsampling parameters
    parser.add_argument("--upsample", type=bool, default=False)
    parser.add_argument("--portion_out", type=float, default=0.5)
    parser.add_argument("--upsample_times", type=int, default=1)
    parser.add_argument("--last_feature_path", type=str, default=None)
    parser.add_argument("--last_model_path", type=str, default=None)

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


def load_model(opt):
    if opt.datasets == "mnist":
        in_channels = 1
    else:
        in_channels = 3

    if opt.ensemble_mode == "end":
        model = SupConResNet_end(name=opt.model, feat_dim=opt.feat_dim, in_channels=in_channels)
    else:
        model = SupConResNet_inter(name=opt.model, feat_dim=opt.feat_dim, in_channels=in_channels)

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

    return model


def normalFeatureReading(data_loader, model, opt):
    outputs_backbone = []
    outputs_head1, outputs_head2, outputs_head3 = [], [], []
    labels = []

    for i, (img, label, _) in enumerate(data_loader):

        print(i)
        if i > opt.break_idx:
            break

        output, output_encoder = model(img), model.encoder(img)
        output_head1, output_head2, output_head3 = output

        outputs_head1.append(np.squeeze(output_head1.detach().numpy()))
        outputs_head2.append(np.squeeze(output_head2.detach().numpy()))
        outputs_head3.append(np.squeeze(output_head3.detach().numpy()))
        outputs_backbone.append(np.squeeze(output_encoder.detach().numpy()))
        labels.append(label.numpy())

    with open(opt.save_path, "wb") as f:
        pickle.dump((outputs_head1, outputs_head2, outputs_head3, outputs_backbone, labels), f)


def meanList(l):
    if len(l) == 0:
        return 0
    else:
        return sum(l) * 1.0 / len(l)


def set_data(opt, class_idx=None):
    if opt.if_train == "train" or opt.if_train == "full":
        datasets = get_train_datasets(opt, class_idx)
    elif opt.if_train == "test_known":
        datasets = get_test_datasets(opt, class_idx)
    elif opt.if_train == "test_unknown":
        datasets = get_outlier_datasets(opt)

    return datasets


def featureMerge(featureList, opt):
    featureMaps_head1 = []
    featureMaps_head2 = []
    featureMaps_head3 = []
    featureMaps_backbone = []
    labels = []
    print(opt.save_path_all)

    for featurePath in featureList:

        with open(featurePath, "rb") as f:
            features_head1, features_head2, features_head3, feature_backbone, labels_part = pickle.load(f)

        featureMaps_head1 = featureMaps_head1 + features_head1
        featureMaps_head2 = featureMaps_head2 + features_head2
        featureMaps_head3 = featureMaps_head3 + features_head3
        featureMaps_backbone = featureMaps_backbone + feature_backbone
        labels = labels + labels_part

    featureMaps_backbone = np.array(featureMaps_backbone, dtype=object)
    featureMaps_head1 = np.array(featureMaps_head1, dtype=object)
    featureMaps_head2 = np.array(featureMaps_head2, dtype=object)
    featureMaps_head3 = np.array(featureMaps_head3, dtype=object)

    featureMaps_backbone = np.squeeze(featureMaps_backbone)
    featureMaps_head1 = np.squeeze(featureMaps_head1)
    featureMaps_head2 = np.squeeze(featureMaps_head2)
    featureMaps_head3 = np.squeeze(featureMaps_head3)
    labels = np.squeeze(np.array(labels))

    with open(opt.save_path_all, 'wb') as f:
        pickle.dump((featureMaps_head1, featureMaps_head2, featureMaps_head3, featureMaps_backbone, labels), f)


if __name__ == "__main__":

    opt = parse_option()

    model = load_model(opt)
    print("Model loaded!!")

    featurePaths = []

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
