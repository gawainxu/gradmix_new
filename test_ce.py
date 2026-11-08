#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 22:02:16 2024

@author: zhi
"""

import os
import sys
BASE_PATH = "/home/sysgen/Jiawen/SupContrast-master"
sys.path.append(BASE_PATH) 

import argparse

import torch
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from torchvision import datasets
import numpy as np
import pickle
import copy
from itertools import chain
from scipy.spatial.distance import mahalanobis

from networks.resnet_big import SupCEResNet

from util import  feature_stats
from util import AverageMeter, accuracy_plain
from distance_utils  import sortFeatures
from datautil import get_test_datasets


torch.multiprocessing.set_sharing_strategy('file_system')

def parse_option():

    parser = argparse.ArgumentParser('argument for feature reading')

    parser.add_argument('--datasets', type=str, default='cifar10',
                        choices=["cifar-10-100-10", "cifar-10-100-50", 'cifar10', "tinyimgnet", 'mnist', "svhn"], help='dataset')
    parser.add_argument('--data_folder', type=str, default=None, help='path to custom dataset')
    parser.add_argument('--model', type=str, default="resnet18")
    parser.add_argument("--model_path", type=str, default="/save/CE/cifar10_models/cifar10_resnet18_1trail_0_128_256/last.pth")
    parser.add_argument("--num_classes", type=int, default=6)
    

    parser.add_argument("--trail", type=int, default=0)
    parser.add_argument("--split_train_val", type=bool, default=True)
    parser.add_argument("--action", type=str, default="testing_known",
                        choices=["training_supcon", "trainging_linear", "testing_known", "testing_unknown", "feature_reading"])
    parser.add_argument('--method', type=str, default='SupCon',
                        choices=['SupCon', 'SimCLR'], help='choose method')
    parser.add_argument("--lr", type=str, default=0.001)
    parser.add_argument("--training_bz", type=int, default=200)
    parser.add_argument("--mem_size", type=int, default=500)
    parser.add_argument("--if_train", type=str, default="train", choices=['train', 'val', 'test_known', 'test_unknown'])
    parser.add_argument('--batch_size', type=int, default=100, help='batch_size')
    parser.add_argument('--num_workers', type=int, default=4, help='num of workers to use')



    opt = parser.parse_args()

    opt = parser.parse_args()
    opt.main_dir = os.getcwd()
    opt.model_path = opt.main_dir + opt.model_path

    return opt


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


def set_model(opt):

    model = SupCEResNet(name=opt.model, num_classes=opt.num_classes)
    model = load_model(model, opt.model_path)
    model.eval()
    model = model.cpu()

    return model


def set_loader(opt):
    # construct data loader
    test_dataset = get_test_datasets(opt) 
    
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=opt.batch_size, shuffle=True,
                                              num_workers=opt.num_workers, pin_memory=True)

    return test_loader


def testing_nn_classifier(model, dataloader):

    model.eval()
    top1 = AverageMeter()

    for idx, (images, labels) in enumerate(dataloader):

        #print(idx)
        #images = images.cuda(non_blocking=True)
        #labels = labels.cuda(non_blocking=True)
        bsz = labels.shape[0]
        output = model(images)
        output = torch.argmax(output, dim=1)

        acc = accuracy_plain(output, labels)
        top1.update(acc, bsz)

    return top1.avg

        

if __name__ == "__main__":
    
    opt = parse_option()

    model = set_model(opt)
    load_model(model, opt.model_path)
    print("Model loaded!!")
    

    data_loader = set_loader(opt)
    avg_accuracy = testing_nn_classifier(model, data_loader)
    print("ID", opt.trail, "Average NN accuracy on inlier testing data is: ", avg_accuracy)
  