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

from networks.ts_models import basic_CNN_multihead
from networks.resnet_multi import SupConResNet_MultiHead
from networks.resnet_big import SupConResNet_MultiHead_remix
from datautil import num_inlier_classes_mapping

from torch.utils.data import DataLoader
from datautil import get_train_datasets, get_test_datasets, get_outlier_datasets, osr_splits_inliers, \
    osr_splits_outliers

torch.multiprocessing.set_sharing_strategy('file_system')

def parse_option():
    parser = argparse.ArgumentParser('argument for feature reading')

    parser.add_argument('--model', type=str, default="basic_cnn_multi")
    parser.add_argument("--model_path", type=str,
                        default="/save/SupCon/paderborn_models/paderborn_basic_cnn_multi_trail_0_128_512_0.5_0.1_0.05_256_vibration/last.pth")
    parser.add_argument("--split_train_val", type=bool, default=True)
    parser.add_argument("--action", type=str, default="feature_reading",
                        choices=["training_supcon", "trainging_linear", "testing_known", "testing_unknown",
                                 "feature_reading"])
    parser.add_argument('--method', type=str, default='SupCon',
                        choices=['SupCon', 'SimCLR'], help='choose method')
    parser.add_argument("--feature_save", type=str, default="/features/")

    # dataset
    parser.add_argument('--datasets', type=str, default='paderborn')
    parser.add_argument('--data_folder', type=str, default=None)
    parser.add_argument("--num_channels", type=int, default=1)
    parser.add_argument("--split_mode", type=str, default='window')
    parser.add_argument('--win_len', type=int, default=4096)
    parser.add_argument('--split_step', type=int, default=100)
    parser.add_argument("--seperate_conditions", type=bool, default=False)
    parser.add_argument("--condition_name", type=str, default="")
    parser.add_argument("--data_channel", type=str, default="vibration")
    parser.add_argument("--trail", type=int, default=0)
    parser.add_argument("--crop_length", type=int, default=4096)
    parser.add_argument("--view_mode", type=str, default="scale_noise")

    # temperature
    parser.add_argument('--temp1', type=float, default=0.05, help='temperature for loss1')
    parser.add_argument('--temp2', type=float, default=0.05, help='temperature for loss2')
    parser.add_argument('--temp3', type=float, default=0.05, help='temperature for loss3')

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


    opt = parser.parse_args()

    opt.main_dir = os.getcwd()
    opt.model_path = opt.main_dir + opt.model_path
    opt.feature_save = opt.main_dir + opt.feature_save

    opt.n_cls = len(osr_splits_inliers[opt.datasets][opt.trail])
    opt.n_outs = len(osr_splits_outliers[opt.datasets][opt.trail])

    if platform.system() == 'Windows':
        opt.model_name = opt.model_path.split("\\")[-2]
    elif platform.system() == 'Linux':
        opt.model_name = opt.model_path.split("/")[-2]
    opt.save_path_all = opt.feature_save + opt.model_name + "_" + str(opt.epoch) + "_" + opt.if_train

    opt.num_classes = num_inlier_classes_mapping[opt.datasets]

    return opt


def load_model(opt):

    if opt.model == "basic_cnn_multi":
        model = basic_CNN_multihead(feature_dim=opt.feat_dim, in_channels=opt.num_channels)
    elif opt.model == "resnet_multi":
        if torch.cuda.device_count() > 1:
            model = SupConResNet_MultiHead_remix(output_dim=opt.out_dim, feat_dim=opt.feat_dim,
                                                 in_channels=opt.num_channels)
        else:
            model = SupConResNet_MultiHead(output_dim=opt.out_dim, feat_dim=opt.feat_dim, in_channels=opt.num_channels)

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
    outputs_head1, outputs_head2, outputs_head3 = [], [], []
    labels = []

    for i, (img, label) in enumerate(data_loader):

        print(i)
        img = img.float()

        output = model(img)
        output_head1, output_head2, output_head3 = output

        outputs_head1.append(np.squeeze(output_head1.detach().numpy()))
        outputs_head2.append(np.squeeze(output_head2.detach().numpy()))
        outputs_head3.append(np.squeeze(output_head3.detach().numpy()))
        labels.append(label.numpy())

    with open(opt.save_path, "wb") as f:
        pickle.dump((outputs_head1, outputs_head2, outputs_head3, labels), f)


def meanList(l):
    if len(l) == 0:
        return 0
    else:
        return sum(l) * 1.0 / len(l)


def set_data(opt):
    if opt.if_train == "train" or opt.if_train == "full":
        datasets = get_train_datasets(opt)
    elif opt.if_train == "test_known":
        datasets = get_test_datasets(opt)
    elif opt.if_train == "test_unknown":
        datasets = get_outlier_datasets(opt)

    return datasets


def featureMerge(featureList, opt):
    featureMaps_head1 = []
    featureMaps_head2 = []
    featureMaps_head3 = []
    labels = []
    print(opt.save_path_all)

    for featurePath in featureList:

        with open(featurePath, "rb") as f:
            features_head1, features_head2, features_head3, labels_part = pickle.load(f)

        featureMaps_head1 = featureMaps_head1 + features_head1
        featureMaps_head2 = featureMaps_head2 + features_head2
        featureMaps_head3 = featureMaps_head3 + features_head3
        labels = labels + labels_part

    featureMaps_head1 = np.array(featureMaps_head1, dtype=object)
    featureMaps_head2 = np.array(featureMaps_head2, dtype=object)
    featureMaps_head3 = np.array(featureMaps_head3, dtype=object)

    featureMaps_head1 = np.squeeze(featureMaps_head1)
    featureMaps_head2 = np.squeeze(featureMaps_head2)
    featureMaps_head3 = np.squeeze(featureMaps_head3)
    labels = np.squeeze(np.array(labels))

    with open(opt.save_path_all, 'wb') as f:
        pickle.dump((featureMaps_head1, featureMaps_head2, featureMaps_head3, labels), f)


if __name__ == "__main__":

    opt = parse_option()

    model = load_model(opt)
    print("Model loaded!!")

    featurePaths = []

    if opt.if_train == "train" or opt.if_train == "test_known" or opt.if_train == "full":
        for r in range(0, opt.n_cls):
            opt.save_path = opt.feature_save + "/temp" + str(r)
            featurePaths.append(opt.save_path)
            datasets = set_data(opt)
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
