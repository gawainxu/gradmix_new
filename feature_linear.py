#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 12 20:07:17 2025

@author: zhi
"""

import os
import sys
import pickle
import argparse

import numpy as np
import torch
from torch.utils.data import Dataset
import torch.optim as optim
from torchvision import transforms

from networks.resnet_big import LinearClassifier
from util import AverageMeter, accuracy


def parse_option():

    parser = argparse.ArgumentParser('argument for feature comparision')

    parser.add_argument("--train_feature_path", type=str, default="/features/cifar10_resnet18_vanilia__SimCLR_1.0_0.0_0.05_trail_0_128_256_600_train", 
                        help="path to the feature file")
    # cifar10_resnet18_vanilia__SimCLR_1.0_1.0_0.05_trail_0_128_256_600_train
    # cifar10_resnet18_vanilia__SimCLR_1.0_0.0_0.05_trail_3_128_256_600_train
    # cifar10_resnet18_mixup_positive_alpha_1.0_beta_1.0_layersaliencymix_2,3,4_SimCLR_1.0_1.2_0.05_trail_1_128_256_twostage_old_augmented_600_train
    parser.add_argument("--test_feature_path", type=str, default="/features/cifar10_resnet18_vanilia__SimCLR_1.0_0.0_0.05_trail_0_128_256_600_test_known", 
                        help="path to the feature file")
    parser.add_argument("--feature_name", type=str, default="head")
    parser.add_argument("--num_classes", type=int, default=6)
    parser.add_argument("--data_task_id", type=int, default=0)
    parser.add_argument("--linear_input_dim", type=int, default=128)
    parser.add_argument("--print_freq", type=int, default=100)
    
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--learning_rate", type=float, default=0.01)
    
    opt = parser.parse_args()
    opt.main_dir = os.getcwd()
    opt.train_feature_path = opt.main_dir + opt.train_feature_path
    opt.test_feature_path = opt.main_dir + opt.test_feature_path
    print(opt.train_feature_path)
    
    return opt


def sortFeatures(mixedFeatures, labels, num_classes, opt):
        
    sortedFeatures = []
    for i in range(num_classes):
        sortedFeatures.append([])

    print("mixedFeatures", np.array(mixedFeatures).shape)
    
    for i, l in enumerate(labels):
        l = l.item()                         
        feature = mixedFeatures[i]
        feature = feature.reshape([-1])
        sortedFeatures[l - opt.data_task_id * opt.num_classes].append(feature)
        
    return sortedFeatures


class features_set(Dataset):
    
    def __init__(self, features, labels, transform=None):
        
        self.features = features
        self.labels = labels
        self.transform = transform
        
    def __len__(self):  
        return len(self.features)
    
    def __getitem__(self, index):       
        feature = torch.from_numpy(self.features[index].astype(np.float32))
        label = torch.from_numpy(np.array(self.labels[index]))
        return feature, label


class avg_features_set(Dataset):
    
    def __init__(self, sorted_features):
        
        self.features = []
        self.labels = []

        for l, features_c in enumerate(sorted_features):
            features_c = np.array(features_c, dtype=np.float32)
            self.features.append(np.mean(features_c, axis=0))
            self.labels.append(l)
            
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, index):
        feature = torch.from_numpy(self.features[index].astype(np.float32))
        label = torch.from_numpy(np.array(self.labels[index]))
        return feature, label
            


def set_model(opt):
    
    criterion = torch.nn.CrossEntropyLoss()
    classifier = LinearClassifier(num_classes=opt.num_classes, 
                                  feat_dim=opt.linear_input_dim)
    classifier = classifier.cuda()
    criterion = criterion.cuda()
    optimizer = optim.Adam(classifier.parameters(), lr=opt.learning_rate)
    
    return classifier, criterion, optimizer


def train(classifier, criterion, optimizer, dataloader, opt):
    
    classifier.train()
    
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    
    for epoch in range(opt.epochs):

        for idx, (data, labels) in enumerate(dataloader):
            #print(labels)
            data = data.cuda()
            labels = labels.cuda()
            output = classifier(data)
            loss = criterion(output, labels)
            
            losses.update(loss.item(), opt.batch_size)
            acc = accuracy(output, labels, topk=(1, opt.num_classes))
            acc1, acc5 = acc[0].item(), acc[1].item()
            top1.update(acc1, opt.batch_size)
            top5.update(acc5, opt.batch_size)

            # SGD
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if (idx + 1) % opt.print_freq == 0:
                print('Train: [{0}][{1}/{2}]\t'
                      'loss {loss.val:.3f} ({loss.avg:.3f})\t'
                      'Acc1 {top1.val:.3f} ({top1.avg:.3f})'
                      'Acc5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                       epoch, idx + 1, len(dataloader), 
                       loss=losses, top1=top1, top5=top5))
                sys.stdout.flush()
                
    return losses, top1, top5


def test(classifier, dataloader, opt):
    
    top1 = AverageMeter()
    
    for idx, (data, label) in enumerate(dataloader):
        data = data.cuda()
        label = label.cuda()
        output = classifier(data)
        #print(output, label)
        acc = accuracy(output, label, topk=(1, opt.num_classes))
        acc1, _ = acc[0].item(), acc[1].item()
        top1.update(acc1, 1)
        
    return top1.avg


if __name__ == "__main__":
    
    opt = parse_option()
    with open(opt.train_feature_path, "rb") as f:
        train_features_head, train_features_backbone, _, train_labels = pickle.load(f)

    with open(opt.test_feature_path, "rb") as f:
        test_features_head, test_features_backbone, _, test_labels = pickle.load(f)
    
    if opt.feature_name == "head":
       train_features = train_features_head
       test_features = test_features_head
    else:
       train_features = train_features_backbone
       test_features = test_features_backbone
        
    train_dataset = features_set(train_features, train_labels)
    #print(type(train_dataset[0][0]), type(train_dataset[0][1]), train_dataset[0][0].shape)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True,
                                                   num_workers=1, pin_memory=True)

    test_dataset = features_set(test_features, test_labels)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False,
                                                  num_workers=1, pin_memory=True)


    classifier, criterion, optimizer = set_model(opt)
    losses, top1, top5 = train(classifier, criterion, optimizer, train_dataloader, opt)
    test_acc = test(classifier, test_dataloader, opt)
    print("original testing accuracy", test_acc)
    
    sorted_train_features = sortFeatures(train_features, train_labels, opt.num_classes, opt)
    sorted_test_features = sortFeatures(test_features, test_labels, opt.num_classes, opt)
    avg_features_set = avg_features_set(sorted_test_features)
    avg_test_dataloader = torch.utils.data.DataLoader(avg_features_set, batch_size=1, shuffle=True,
                                                      num_workers=1, pin_memory=True)
    avg_test_acc = test(classifier, avg_test_dataloader, opt)
    print("average testing accuracy", avg_test_acc)

