#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 28 10:38:45 2024

@author: zhi
"""

import pickle
import os
import argparse

import torch
import numpy as np
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torchvision.transforms.functional import to_pil_image
import matplotlib.pyplot as plt

from networks.resnet_big import SupConResNet, LinearClassifier
from main_testing import load_model
from losses import SupConLoss
from datautil import get_gradcam_datasets, get_train_datasets, osr_splits_inliers
from torchvision import transforms
from util import TwoCropTransform


def parse_option():
    
    parser = argparse.ArgumentParser('argument for grad cam')
    parser.add_argument('--datasets', type=str, default='tinyimgnet',
                        choices=["cifar-10-100-10", "cifar-10-100-50", 'cifar10', "tinyimgnet", 'mnist', "svhn"], help='dataset')
    parser.add_argument("--num_classes", type=int, default=20)
    parser.add_argument("--trail", type=int, default=0)

    # for "single" and "sim"
    parser.add_argument("--class_idx", type=int, default=0)
    parser.add_argument("--data_id", type=int, default=30)
    # for "supcon"
    parser.add_argument("--class_idxs", type=list, default=[4])
    parser.add_argument("--data_ids", type=list, default=[[123,231,56,23, 30, 84], [65,67,23,87, 90, 100]])
    parser.add_argument("--img_size", type=int, default=64)
    parser.add_argument("--bsz", type=int, default=256)
    parser.add_argument("--action", type=str, default="training_supcon",
                        choices=["training_supcon", "trainging_linear", "testing_known", "testing_unknown", "feature_reading"])  
    parser.add_argument("--randaug", type=int, default=0)
    parser.add_argument("--augmix", type=bool, default=False)
    
    parser.add_argument('--model', type=str, default="resnet18")
    parser.add_argument("--model_path", type=str, default="/save/SupCon/tinyimgnet_models/tinyimgnet_resnet18_original_data__vanilia__SimCLR_1.0_1.0_0.05_trail_0_128_256/ckpt_epoch_600.pth")
    parser.add_argument('--temp', type=float, default=0.05, help='temperature for loss function')
    
    parser.add_argument("--mode", type=str, default="sim", choices=["single", "supcon", "sim"], help="Mode of the loss function that used to compute the loss")
    parser.add_argument("--grad_layers", type=str, default="3")
    parser.add_argument("--threshold", type=float, default=1e-3)
    
    opt = parser.parse_args()
    opt.main_dir = os.getcwd()
    
    opt.model_name = opt.model_path.split("/")[-2]

    opt.model_path = opt.main_dir + opt.model_path
    print(opt.model_path)
    opt.class_id = osr_splits_inliers[opt.datasets][opt.trail][opt.class_idx]
    
    opt.save_path = "./featuremaps/" + opt.model_name
                
    layers = opt.grad_layers.split(",")
    opt.grad_layers = list([])
    for l in layers:
        opt.grad_layers.append(int(l))
        
    return opt


# global variables
global gradients
gradients = {}

global activations
activations = {}



def backward_hook(module, grad_input, grad_output):
    
    #print("Backward Hook Running", grad_output[0].shape)
    gradients[str(4-grad_output[0].shape[-1]/8)] = grad_output
    
    
def forward_hook(module, input, output):
    
    #print("Forward Hook Running", output[0].shape)
    activations[str(4-output[0].shape[-1]/8)] = output
    
    

if __name__ == "__main__":
    
    opt = parse_option()
    
    # load data and model
    model = SupConResNet(name=opt.model)
    model = load_model(model, path=opt.model_path)
    model.eval()
    model = model.cuda()
    
    dataset = get_train_datasets(opt)
    dataset_cam = get_gradcam_datasets(opt)
    criterion = SupConLoss(temperature=opt.temp)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=opt.bsz, shuffle=False,
                                              num_workers=4, pin_memory=True, drop_last=True)
    
    # read gradients and feature maps
    hooks = []
    hooks.append(model.encoder.layer3[-1].register_forward_hook(hook=forward_hook))
    hooks.append(model.encoder.layer3[-1].register_full_backward_hook(hook=backward_hook)) 
    
    # compute the statistics of the activation maps
    sum_of_cams = 0
    for idx, (images, labels) in enumerate(data_loader):
        
        images1 = images[0]
        images2 = images[1]
        images1 = images1.cuda()
        images2 = images2.cuda()
        images = torch.cat([images1, images2], dim=0)
        
        features = model(images)
        features1, features2 = torch.split(features, [256, 256], dim=0)
        features = torch.cat([features1.unsqueeze(1), features2.unsqueeze(1)], dim=1)
        loss1 = criterion(features, labels)
        loss2 = criterion(features)          # torch.matmul(f1, f2.T)  #   labels
        loss = loss1 + loss2
        loss.backward()
    
        # activation maps of one batch
        cam = torch.zeros((opt.bsz, 1, 16, 16), device=torch.device('cuda'))   #
        
        for key in activations.keys():
            
            activations_layer = activations[key]
            gradients_layer = gradients[key][0]
            
            for i in range(opt.bsz):
                
                activation_i = activations_layer[i]
                gradient_i = gradients_layer[i]
                with torch.no_grad():
                    activation_maps = activation_i * F.relu(gradient_i)
                    cam_layer = torch.sum(activation_maps, dim=0).unsqueeze(0).unsqueeze(0)
                    cam[i] = cam[i] + cam_layer #F.interpolate(cam_layer, size=(opt.img_size, opt.img_size), mode="bilinear", align_corners=False)[0]
                
        sum_of_cams += torch.count_nonzero(cam > opt.threshold)
    
    print("sum of non-zero cams", sum_of_cams / len(dataset))
    with open(opt.save_path, "wb") as f:
        pickle.dump((gradients, activations), f)
                