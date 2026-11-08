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
import inspect

import torch
import pickle

from networks.resnet_big import SupConResNet, LinearClassifier
from networks.vgg import SupConVGG
from networks.simCNN import simCNN_contrastive
from networks.mlp import SupConMLP
from networks.resnet_big import ResNet_Pretrain
from featureMerge import featureMerge
from datautil import num_inlier_classes_mapping
from  data_loader import safe_collate

from torch.utils.data import DataLoader, Subset
from datautil import get_train_datasets, get_test_datasets, get_outlier_datasets, osr_splits_inliers, osr_splits_outliers

torch.multiprocessing.set_sharing_strategy('file_system')


breaks = {"cifar-10-100-10": {"train": 5000, "test_known":500, "test_unknown": 50, "full": 100000}, 
          "cifar-10-100-50": {"train": 5000, "test_known": 500, "test_unknown": 50, "full": 100000},
          "cifar100": {"train": 5000, "test_known": 500, "test_unknown": 50, "full": 100000},
           'cifar10': {"train": 5000, "test_known": 500, "test_unknown": 500, "full": 100000},
           "tinyimgnet": {"train": 5000, "test_known": 100, "test_unknown": 20, "full": 100000},
           'mnist': {"train": 5000, "test_known": 500, "test_unknown": 500, "full": 100000},
           "svhn": {"train": 5000, "test_known": 500, "test_unknown": 500, "full": 100000},
           "cub": {"train": 5000, "test_known": 500, "test_unknown": 500, "full": 100000},
           "cars": {"train": 5000, "test_known": 500, "test_unknown": 500, "full": 100000},
           "aircraft": {"train": 5000, "test_known": 500, "test_unknown": 500, "full": 100000},
           "imagenet100": {"train": 5000, "test_known": 1000, "test_unknown": 10000, "full": 100000},
           "imagenet1k": {"train": 200, "test_known": 500, "test_unknown": 500, "full": 100000},
           "FUB": {"train": 5000, "test_known": 500, "test_unknown": 500, "full": 100000},}

def parse_option():

    parser = argparse.ArgumentParser('argument for feature reading')

    parser.add_argument('--datasets', type=str, default='cars', help='dataset')
    parser.add_argument('--data_folder', type=str, default=None, help='path to custom dataset')
    parser.add_argument('--model', type=str, default="resnet50_pretrain", choices=["resnet18", "vgg16", "resnet50_pretrain", "simCNN", "MLP"])
    parser.add_argument("--model_path", type=str,
                        default="/save/SupCon/cars_models/cars_resnet50_pretrain_original_data__mixup_positive_alpha_1.0_beta_1.0_layersaliencymix_0,1,2,3_Joint_0.5_0.5_trail_0_128_256_split_128/last.pth")
    parser.add_argument("--linear_model_path", type=str, default=None)
    parser.add_argument("--trail", type=int, default=0)
    parser.add_argument("--trail_outliers", type=int, default=0)
    parser.add_argument("--split_train_val", type=bool, default=True)
    parser.add_argument("--start_class", type=int, default=0)
    parser.add_argument("--action", type=str, default="feature_reading",
                        choices=["training_supcon", "trainging_linear", "testing_known", "testing_unknown", "feature_reading"])
    parser.add_argument('--method', type=str, default='SupCon',
                        choices=['SupCon', 'SimCLR'], help='choose method')
    parser.add_argument("--feature_save", type=str, default="/features/")
    parser.add_argument("--if_merge", type=bool, default=False)

    # temperature
    parser.add_argument('--temp', type=float, default=0.05, help='temperature for loss')
    parser.add_argument('--temp1', type=float, default=0.05, help='temperature for loss function late')
    parser.add_argument('--temp2', type=float, default=0.01, help='temperature for loss function early')
    parser.add_argument("--lam", type=float, default=1.0)

    parser.add_argument("--epoch", type=int, default = 600)
    parser.add_argument("--tau_strategy", type=str, default="fixed", choices=["fixed", "fixed_set", "fixed_set_diff", "cosine", "linear", "exp"])
    parser.add_argument("--cosine_period", type=float, default=1.0)
    parser.add_argument("--augmentation_method", type=str, default="vanilia", choices=["vanilia", "upsampling", "mixup"])
    parser.add_argument("--architecture", type=str, default="single", choices=["single", "multi"])
    parser.add_argument("--ensemble_num", type=int, default=1)
    parser.add_argument("--feat_dim", type=int, default=128)

    parser.add_argument("--lr", type=str, default=0.01)
    parser.add_argument("--training_bz", type=int, default=600)
    parser.add_argument("--if_train", type=str, default="test_known", choices=['train', 'val', 'test_known', 'test_unknown', "full"])
    parser.add_argument('--batch_size', type=int, default=1, help='batch_size')
    parser.add_argument('--num_workers', type=int, default=4, help='num of workers to use')

    # upsampling parameters
    parser.add_argument("--upsample", type=bool, default=False)
    parser.add_argument("--portion_out", type=float, default=0.5)
    parser.add_argument("--upsample_times", type=int, default=1)
    parser.add_argument("--last_feature_path", type=str, default=None)
    parser.add_argument("--last_model_path", type=str, default=None)

    # mixup parameters
    parser.add_argument("--alpha_negative", type=float, default=0.2, help="between 0.2 to 0.4")
    parser.add_argument("--alpha_positive", type=float, default=0.2, help="between 0.2 to 0.4")
    parser.add_argument("--intra_inter_mix_positive", type=bool, default=True, help="intra=True, inter=False")
    parser.add_argument("--intra_inter_mix_negative", type=bool, default=True, help="intra=True, inter=False")
    parser.add_argument("--mixup_positive", type=bool, default=False)
    parser.add_argument("--mixup_negative", type=bool, default=False)
    parser.add_argument("--p", type=float, default=1.0)
    parser.add_argument("--positive_method", type=str, default="no", choices=["min_similarity", "random", "prob_similarity", "no"])
    parser.add_argument("--negative_method", type=str, default="no", choices=["max_similarity", "random", "no"])


    opt = parser.parse_args()

    opt.main_dir = os.getcwd()
    opt.model_path = opt.main_dir + opt.model_path
    opt.feature_save = opt.main_dir + opt.feature_save
    if opt.linear_model_path is not None:
        opt.linear_model_path = opt.main_dir + opt.linear_model_path

    opt.n_cls = len(osr_splits_inliers[opt.datasets][opt.trail])
    opt.n_outs = len(osr_splits_outliers[opt.datasets][opt.trail_outliers])

    opt.break_idx = breaks[opt.datasets][opt.if_train]
    if platform.system() == 'Windows':
        opt.model_name = opt.model_path.split("\\")[-2]
    elif platform.system() == 'Linux':
        opt.model_name = opt.model_path.split("/")[-2]
    opt.save_path_all = (opt.feature_save + opt.model_name + "_" + str(opt.epoch)
                         + "_outliers_" + str(opt.trail_outliers) + "_" + opt.if_train)

    opt.num_classes = num_inlier_classes_mapping[opt.datasets]

    return opt


def load_model(opt):

    if opt.datasets == "mnist":
        in_channels = 1
    elif opt.datasets == "FUB":
        in_channels = 1
    else:
        in_channels = 3

    if opt.model == "resnet18" or opt.model == "resnet34":
        model = SupConResNet(name=opt.model, feat_dim=opt.feat_dim, in_channels=in_channels)
    elif opt.model == "MLP":
        model = SupConMLP(feat_dim=opt.feat_dim)
    elif opt.model == "resnet50_pretrain":
        model = ResNet_Pretrain(name='resnet50', feat_dim=opt.feat_dim)
    elif opt.model in ["vgg16", "vgg11", "vgg_s_bn"]:
        model = SupConVGG(name=opt.model, feat_dim=opt.feat_dim, in_channels=in_channels)
    else:
        model = simCNN_contrastive(opt, feature_dim=opt.feat_dim, in_channels=in_channels)
    ckpt = torch.load(opt.model_path, map_location='cpu')
    state_dict = ckpt['model']

    new_state_dict = {}
    for k, v in state_dict.items():
        k = k.replace("module.", "")
        new_state_dict[k] = v

    state_dict = new_state_dict
    model = model.cuda()
    model.load_state_dict(state_dict)
    model.eval()


    if opt.linear_model_path is not None:

        linear_model = LinearClassifier(name=opt.model, num_classes=opt.n_cls)
        ckpt = torch.load(opt.linear_model_path, map_location='cpu')
        state_dict = ckpt['model']
        linear_model = linear_model.cpu()
        linear_model.load_state_dict(state_dict)

        """
        
        new_state_dict = {}
        for k, v in state_dict.items():
            k = k.replace("module.", "")
            new_state_dict[k] = v

        state_dict = new_state_dict
        linear_model = linear_model.cpu()
        linear_model.load_state_dict(state_dict)
        """

        linear_model.eval()

        return model, linear_model

    else:
        return model, None


def normalFeatureReading(data_loader, model, linear_model, opt):
    
    outputs_backbone = []
    outputs = []
    outputs_linear = []
    labels = []

    for i, (img, label) in enumerate(data_loader):

        if img is None:
            i -= 1
            continue
        print(i)
        if i > opt.break_idx:
            break

        img = img.cuda()
        output, output_encoder = model(img)[0], model.encoder(img)

        if linear_model is not None:
            linear_output = linear_model(model.encoder(img))
            outputs.append(output.detach().numpy())
            outputs_backbone.append(output_encoder[-1].detach().numpy())
            outputs_linear.append(linear_output.detach().numpy())
        else:
            outputs.append(output.cpu().detach().numpy())
            outputs_backbone.append(output_encoder[-1].cpu().detach().numpy())

        labels.append(label.numpy())

    with open(opt.save_path, "wb") as f:
        pickle.dump((outputs, outputs_backbone, outputs_linear, labels), f)
        
            
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

    model, linear_model = load_model(opt)
    print("Model loaded!!")
    
    featurePaths= []

    if opt.if_train == "train" or opt.if_train == "test_known" or opt.if_train == "full":
        for r in range(0, opt.n_cls):
            print(r)
            opt.save_path = opt.feature_save + "temp" + str(r)
            featurePaths.append(opt.save_path)
            #if r < opt.start_class or r > opt.start_class + 1:
            #    continue
            datasets = set_data(opt, class_idx=r)

            dataloader = DataLoader(datasets, batch_size=1, shuffle=False, sampler=None,
                                    num_workers=1, collate_fn=safe_collate)
            normalFeatureReading(dataloader, model, linear_model, opt)

        featureMerge(featurePaths, opt.save_path_all)

    else:
        """
         for r in range(0, opt.n_outs):                            
            opt.save_path = opt.feature_save + "temp" + str(r)
            featurePaths.append(opt.save_path)
            datasets = set_data(opt, class_idx=r)
            dataloader = DataLoader(datasets, batch_size=1, shuffle=False, sampler=None, 
                                    num_workers=1)
            normalFeatureReading(dataloader, model, linear_model, opt)

         featureMerge(featurePaths, opt)
         """
        datasets = set_data(opt)
        dataloader = DataLoader(datasets, batch_size=1, shuffle=False, sampler=None,
                                num_workers=1, collate_fn=safe_collate)
        opt.save_path = opt.save_path_all
        normalFeatureReading(dataloader, model, linear_model, opt)
