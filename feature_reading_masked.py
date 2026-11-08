#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 14 15:29:30 2022

@author: zhi
"""

import os
import platform
import sys
BASE_PATH = "/home/sysgen/Jiawen/comprehensive_OSR"
sys.path.append(BASE_PATH) 

import argparse

import torch
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import numpy as np
import pickle
from itertools import chain

from networks.resnet_big import SupConResNet, LinearClassifier
from networks.resnet_preact import SupConpPreactResNet
from networks.simCNN import simCNN_contrastive
from networks.mlp import SupConMLP
from featureMerge import featureMerge

from torch.utils.data import DataLoader
from masked_datasets import my_VOC_subsects

torch.multiprocessing.set_sharing_strategy('file_system')


def parse_option():

    parser = argparse.ArgumentParser('argument for feature reading')

    parser.add_argument('--datasets', type=str, default='imaget100_voc',
                        choices=["imaget100_voc", 'imaget100_s'], help='dataset')
    parser.add_argument('--data_folder', type=str, default=None, help='path to custom dataset')
    parser.add_argument('--model', type=str, default="resnet18", choices=["resnet18", "resnet34", "preactresnet18", "preactresnet34", "simCNN", "MLP"])
    parser.add_argument("--data_root", type=str, default = "../datasets")
    parser.add_argument("--model_path", type=str, default="/save/SupCon/imagenet100_models/imagenet100_resnet18_original_data__vanilia__SimCLR_0.0_1.0_0.05_trail_0_128_256/ckpt_epoch_100.pth")
    parser.add_argument("--linear_model_path", type=str, default=None)
    parser.add_argument("--trail", type=int, default=3)
    parser.add_argument('--method', type=str, default='SupCon',
                        choices=['SupCon', 'SimCLR'], help='choose method')
    parser.add_argument("--feature_save", type=str, default="/features/")
    parser.add_argument("--feat_dim", type=int, default=128)


    opt = parser.parse_args()

    opt.main_dir = os.getcwd()
    opt.model_path = opt.main_dir + opt.model_path
    opt.feature_save = opt.main_dir + opt.feature_save
    if opt.linear_model_path is not None:
        opt.linear_model_path = opt.main_dir + opt.linear_model_path

    if platform.system() == 'Windows':
        opt.model_name = opt.model_path.split("\\")[-2]
    elif platform.system() == 'Linux':
        opt.model_name = opt.model_path.split("/")[-2]
    opt.save_path_all = opt.feature_save + opt.model_name + "_" + opt.datasets 

    return opt


def load_model(opt):

    in_channels = 3

    if opt.model == "resnet18" or opt.model == "resnet34":
        model = SupConResNet(name=opt.model, feat_dim=opt.feat_dim, in_channels=in_channels)
    elif opt.model == "preactresnet18" or opt.model == "preactresnet34":
        model = SupConpPreactResNet(name=opt.model, feat_dim=opt.feat_dim, in_channels=in_channels)
    elif opt.model == "MLP":
        model = SupConMLP(feat_dim=opt.feat_dim)
    else:
        model = simCNN_contrastive(opt,  feature_dim=opt.feat_dim, in_channels=in_channels)
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
        
        print(i)
        if i > opt.break_idx:
            break

        if opt.method == "SupCon":
            output, output_encoder = model(img)[0], model.encoder(img)
        else:
            output = model.encoder(img)

        if linear_model is not None:
            linear_output = linear_model(model.encoder(img))
            outputs.append(output.detach().numpy())
            outputs_backbone.append(output_encoder[-1].detach().numpy())
            outputs_linear.append(linear_output.detach().numpy())
        else:
            outputs.append(output.detach().numpy())
            outputs_backbone.append(output_encoder[-1].detach().numpy())

        labels.append(label.numpy())

    with open(opt.save_path, "wb") as f:
        pickle.dump((outputs, outputs_backbone, outputs_linear, labels), f)
        
            
def meanList(l):
    
    if len(l) == 0:
        return 0
    else:
        return sum(l)*1.0 / len(l)


def set_data(opt):

    transform = transforms.Compose([transforms.RandomResizedCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.485, 0.456, 0.406),
                                                         (0.229, 0.224, 0.225))])
    datasets = my_VOC_subsects(root=opt.data_root, transform=transform)
    data_loader = DataLoader(datasets, batch_size=1, shuffle=False,
                             num_workers=1, pin_memory=True)
    return data_loader

if __name__ == "__main__":
    
    opt = parse_option()

    model, linear_model = load_model(opt)
    print("Model loaded!!")
    
    data_loader = set_data(opt)
    #normalFeatureReading(data_loader, model, linear_model, opt)

       
