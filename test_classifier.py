#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 28 16:16:13 2024

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

from networks.resnet_big import SupConResNet, LinearClassifier

from util import  feature_stats
from util import accuracy, AverageMeter, accuracy_plain, AUROC, OSCR, down_sampling
from distance_utils  import sortFeatures
from datautil import get_test_datasets, get_outlier_datasets


torch.multiprocessing.set_sharing_strategy('file_system')

def parse_option():

    parser = argparse.ArgumentParser('argument for feature reading')

    parser.add_argument('--datasets', type=str, default='cifar10',
                        choices=["cifar-10-100-10", "cifar-10-100-50", 'cifar10', 'cifar100', "tinyimgnet", 'mnist', "svhn"], help='dataset')
    parser.add_argument('--data_folder', type=str, default=None, help='path to custom dataset')
    parser.add_argument('--model', type=str, default="resnet18")
    parser.add_argument("--model_path", type=str, default="/save/")
    parser.add_argument("--linear_model_path", type=str, default=None)
    parser.add_argument("--num_classes", type=int, default=200)
    
    parser.add_argument("--exemplar_features_path", type=str, default="/features/curruption_train_tinyimgnet")
    parser.add_argument("--testing_known_features_path", type=str, default="/features/curruption_brightness_2_tinyimgnet")


    parser.add_argument("--trail", type=int, default=0)
    parser.add_argument("--split_train_val", type=bool, default=True)
    parser.add_argument("--action", type=str, default="testing_known",
                        choices=["training_supcon", "trainging_linear", "testing_known", "testing_unknown", "feature_reading"])
    parser.add_argument('--method', type=str, default='SupCon',
                        choices=['SupCon', 'SimCLR'], help='choose method')
    parser.add_argument("--temp", type=str, default = 0.5)
    parser.add_argument("--lr", type=str, default=0.001)
    parser.add_argument("--training_bz", type=int, default=200)
    parser.add_argument("--mem_size", type=int, default=500)
    parser.add_argument("--if_train", type=str, default="train", choices=['train', 'val', 'test_known', 'test_unknown'])
    parser.add_argument('--batch_size', type=int, default=1, help='batch_size')
    parser.add_argument('--num_workers', type=int, default=4, help='num of workers to use')
    parser.add_argument("--downsampling_ratio_known", type=int, default=10)

    parser.add_argument("--K", type=int, default=3)

    parser.add_argument("--auroc_save_path", type=str, default="./plots/auroc")

    parser.add_argument("--downsample", type=bool, default=False)
    parser.add_argument("--last_feature_path", type=str, default=None)
    parser.add_argument("--downsample_ratio", type=float, default=0.1)
    parser.add_argument("--downsample_ratio_center", type=float, default=0.05)

    opt = parser.parse_args()

    opt = parser.parse_args()
    opt.main_dir = os.getcwd()
    opt.model_path = opt.main_dir + opt.model_path
  
    if opt.linear_model_path is not None:
        opt.linear_model_path = opt.main_dir + opt.linear_model_path

    opt.exemplar_features_path = opt.main_dir + opt.exemplar_features_path
    opt.testing_known_features_path = opt.main_dir + opt.testing_known_features_path
    opt.auroc_save_path = opt.main_dir + opt.auroc_save_path

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

    model = SupConResNet(name=opt.model)
    model = load_model(model, opt.model_path)
    model.eval()
    model = model.cpu()

    if opt.linear_model_path is not None:
        linear_model = LinearClassifier(name=opt.model, num_classes=opt.num_classes, emsembles=opt.ensembles)
        ckpt = torch.load(opt.linear_model_path, map_location='cpu')
        #print(ckpt.keys())
        state_dict = ckpt['model']
        linear_model.load_state_dict(state_dict)
        linear_model = linear_model.cpu()
        linear_model.eval()

        return model, linear_model

    return model


def set_loader(opt):
    # construct data loader
    test_dataset = get_test_datasets(opt)

    if opt.with_outliers:
        outlier_dataset = get_outlier_datasets(opt)
        test_dataset = torch.utils.data.ConcatDataset([test_dataset, outlier_dataset])   
    
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=opt.batch_size, shuffle=True,
                                              num_workers=opt.num_workers, pin_memory=True)

    return test_loader


def testing_nn_classifier(models, classifier, dataloader):

    for model in models:
        model.eval()
    classifier.eval()

    top1 = AverageMeter()

    for idx, (images, labels) in enumerate(dataloader):

        bsz = labels.shape[0]

        features = torch.empty((bsz, 0), dtype=torch.float32)
        for model in models:
            feature = model.encoder(images)
            features = torch.cat((features, feature), dim=1)
        output = classifier(features)

        acc, _ = accuracy(output, labels)
        top1.update(acc, bsz)

    return top1.avg



def KNN_logits(testing_features, sorted_exemplar_features):

    testing_similarity_logits = []

    for idx, testing_feature in enumerate(testing_features):
        similarity_logits = []
        for training_features_c in sorted_exemplar_features:
            
            training_features_c = np.array(training_features_c, dtype=float)
            similarities = np.matmul(training_features_c, testing_feature) / np.linalg.norm(training_features_c, axis=1) / np.linalg.norm(testing_feature)
            ind = np.argsort(similarities)[-opt.K:]
            top_k_similarities = similarities[ind]
            similarity_logits.append(np.sum(top_k_similarities))

        testing_similarity_logits.append(similarity_logits)
    
    testing_similarity_logits = np.array(testing_similarity_logits)
    testing_similarity_logits = np.divide(testing_similarity_logits.T, np.sum(testing_similarity_logits, axis=1)).T                         # normalization

    return testing_similarity_logits




def KNN_classifier(testing_features, testing_labels, sorted_training_features):

    testing_similarity_logits = KNN_logits(testing_features, sorted_training_features)
    prediction_logits, predictions = np.amax(testing_similarity_logits, axis=1), np.argmax(testing_similarity_logits, axis=1)

    acc = accuracy_plain(predictions, testing_labels)

    return prediction_logits, predictions, acc



def feature_classifier(opt):

    with open(opt.exemplar_features_path, "rb") as f:
        features_exemplar_backbone, _, _, labels_examplar = pickle.load(f)         
        features_exemplar_backbone = np.squeeze(np.array(features_exemplar_backbone))

    sorted_exemplar_features = sortFeatures(features_exemplar_backbone, labels_examplar, opt)


    if  opt.testing_known_features_path is not None:
        with open(opt.testing_known_features_path, "rb") as f:
            features_testing_known_backbone, _, _, labels_testing_known = pickle.load(f)           
            features_testing_known_backbone = np.squeeze(np.array(features_testing_known_backbone))
            labels_testing_known = np.squeeze(np.array(labels_testing_known))
    

    features_testing_known_backbone, labels_testing_known = down_sampling(features_testing_known_backbone, labels_testing_known, opt.downsampling_ratio_known)
    prediction_logits_known, predictions_known, acc_known = KNN_classifier(features_testing_known_backbone, labels_testing_known, sorted_exemplar_features)
    
    
    return acc_known

        

if __name__ == "__main__":
    
    opt = parse_option()

    #models, linear_model = set_model(opt)
    #print("Model loaded!!")
    
    acc = feature_classifier(opt)
    print(opt.testing_known_features_path)
    print(acc)
  
