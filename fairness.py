#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 24 23:21:23 2024

@author: zhi
"""


import pickle
import numpy as np

import torch
import argparse
from networks.resnet_big import SupConResNet


def parse_option():
    
    parser = argparse.ArgumentParser('argument for testing')
    parser.add_argument("--train_feature_path", type=str, default="")
    parser.add_argument("--test_know_feature_path", type=str, default="")
    parser.add_argument("--test_unknow_feature_path", type=str, default="")
    
    parser.add_argument("--model", type=str, default="resnet34")
    parser.add_argument("--linear_model_path", type=str, default="")
    parser.add_argument("--model_path", type=str, default="")

    parser.add_argument("--dataset", type=str, default="cifar10")
    
    opt = parser.parse_args()
    
    return opt
    
    

def load_model(opt):
    
    model = SupConResNet(name=opt.model)
    ckpt = torch.load(opt.last_model_path, map_location='cpu')
    state_dict = ckpt['model']
    
    new_state_dict = {}
    for k, v in state_dict.items():
        k = k.replace("module.", "")
        new_state_dict[k] = v

    state_dict = new_state_dict
    model.load_state_dict(state_dict)
    model.eval()
    
    return model


def load_data()