#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 14 15:29:30 2022

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
from datautil import get_test_datasets, get_outlier_datasets, feature_stats, mahalanobis_group


torch.multiprocessing.set_sharing_strategy('file_system')

def parse_option():

    parser = argparse.ArgumentParser('argument for feature reading')

    parser.add_argument('--datasets', type=str, default='cifar10',
                        choices=["cifar-10-100-10", "cifar-10-100-50", 'cifar10', "tinyimgnet", 'mnist', "svhn"], help='dataset')
    parser.add_argument('--data_folder', type=str, default=None, help='path to custom dataset')
    parser.add_argument('--model', type=str, default="resnet18")
    parser.add_argument("--model_path", type=str, default="/save/")
    parser.add_argument("--model_path1", type=str, default=None)
    parser.add_argument("--model_path2", type=str, default=None)
    parser.add_argument("--ensembles", type=int, default=1)
    parser.add_argument("--linear_model_path", type=str, default=None)
    parser.add_argument("--num_classes", type=int, default=10)
    
    parser.add_argument("--exemplar_features_path", type=str, default="/features/")
    parser.add_argument("--testing_known_features_path", type=str, default="/features/")
    parser.add_argument("--testing_unknown_features_path", type=str, default="/features/")

    parser.add_argument("--exemplar_features_path1", type=str, default=None)
    parser.add_argument("--testing_known_features_path1", type=str, default=None)
    parser.add_argument("--testing_unknown_features_path1", type=str, default=None)

    parser.add_argument("--exemplar_features_path2", type=str, default=None)
    parser.add_argument("--testing_known_features_path2", type=str, default=None)
    parser.add_argument("--testing_unknown_features_path2", type=str, default=None)

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
    parser.add_argument("--downsampling_ratio_unknown", type=int, default=20)

    parser.add_argument("--K", type=int, default=100)

    parser.add_argument("--auroc_save_path", type=str, default="./plots/auroc")

    parser.add_argument("--with_outliers", type=bool, default=False)
    parser.add_argument("--downsample", type=bool, default=False)
    parser.add_argument("--last_feature_path", type=str, default=None)
    parser.add_argument("--downsample_ratio", type=float, default=0.1)
    parser.add_argument("--downsample_ratio_center", type=float, default=0.05)

    opt = parser.parse_args()

    opt = parser.parse_args()
    opt.main_dir = os.getcwd()
    opt.model_path = opt.main_dir + opt.model_path
    if opt.model_path1 is not None:
        opt.model_path1 = opt.main_dir + opt.model_path1

    if opt.model_path2 is not None:
        opt.model_path2 = opt.main_dir + opt.model_path2

    if opt.linear_model_path is not None:
        opt.linear_model_path = opt.main_dir + opt.linear_model_path

    opt.exemplar_features_path = opt.main_dir + opt.exemplar_features_path
    opt.testing_known_features_path = opt.main_dir + opt.testing_known_features_path
    opt.testing_unknown_features_path = opt.main_dir + opt.testing_unknown_features_path
    opt.auroc_save_path = opt.main_dir + opt.auroc_save_path

    if opt.exemplar_features_path1 is not None:
        opt.exemplar_features_path1 = opt.main_dir + opt.exemplar_features_path1
    if opt.testing_known_features_path1 is not None:
        opt.testing_known_features_path1 = opt.main_dir + opt.testing_known_features_path1
    if opt.testing_unknown_features_path1 is not None:
        opt.testing_unknown_features_path1 = opt.main_dir + opt.testing_unknown_features_path1

    if opt.exemplar_features_path2 is not None:
        opt.exemplar_features_path2 = opt.main_dir + opt.exemplar_features_path2
    if opt.testing_known_features_path2 is not None:
        opt.testing_known_features_path2 = opt.main_dir + opt.testing_known_features_path2
    if opt.testing_unknown_features_path2 is not None:
        opt.testing_unknown_features_path2 = opt.main_dir + opt.testing_unknown_features_path2

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
    models = []
    models.append(model)

    if opt.model_path1 is not None:
        model1 = copy.deepcopy(model)
        model1 = load_model(model1, opt.model_path1)
        models.append(model1)

    if opt.model_path2 is not None:
        model2 = copy.deepcopy(model)
        model2 = load_model(model2, opt.model_path2)
        models.append(model2)

    if opt.linear_model_path is not None:
        linear_model = LinearClassifier(name=opt.model, num_classes=opt.num_classes, emsembles=opt.ensembles)
        ckpt = torch.load(opt.linear_model_path, map_location='cpu')
        #print(ckpt.keys())
        state_dict = ckpt['model']
        linear_model.load_state_dict(state_dict)
        linear_model = linear_model.cpu()
        linear_model.eval()

        return models, linear_model

    return models


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

        #print(idx)
        #images = images.cuda(non_blocking=True)
        #labels = labels.cuda(non_blocking=True)
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
        #print(idx)
        similarity_logits = []
        for training_features_c in sorted_exemplar_features:
            
            training_features_c = np.array(training_features_c, dtype=float)
            #print("training_features_c", type(training_features_c), training_features_c.shape)
            #print("feature norm, ", np.linalg.norm(training_features_c))
            #print("feature norm, ", np.linalg.norm(testing_feature))
            #print("training_features_c", training_features_c.shape, "testing_feature", testing_feature.shape)
            similarities = np.matmul(training_features_c, testing_feature) / np.linalg.norm(training_features_c, axis=1) / np.linalg.norm(testing_feature)
            ind = np.argsort(similarities)[-opt.K:]
            top_k_similarities = similarities[ind]
            similarity_logits.append(np.sum(top_k_similarities))

        testing_similarity_logits.append(similarity_logits)
    
    testing_similarity_logits = np.array(testing_similarity_logits)
    testing_similarity_logits = np.divide(testing_similarity_logits.T, np.sum(testing_similarity_logits, axis=1)).T                         # normalization

    return testing_similarity_logits


def distances(stats, test_features, mode="mahalanobis"):

    dis_logits_out = []
    dis_logits_in = []
    dis_preds = []
    for features in test_features:
        diss = []
        for i, (mu, var) in enumerate(stats):
            #mu, var = stats[0]                             ##### delete
            if mode == "mahalanobis":
                features_normalized = features - mu
                #dis =  np.matmul(features_normalized, np.linalg.inv(var))
                #dis = np.matmul(dis, np.swapaxes(features_normalized, 0, 1))
                #dis = dis[0][0]
                dis = mahalanobis(features, mu, np.linalg.inv(var))
            else:
                features = np.squeeze(np.array(features))
                dis = features - mu
                dis = np.sum(np.abs(dis))

            diss.append(dis)
        
        dis_logits_out.append(np.min(np.array(diss))/np.sum(np.array(diss)))                   #  !!!!!!!!!!!!!!!!!! minus here !!!!!!!!!!!! to entsprechen 0 for outliers and 1 for inliers, unknown logits, flip for known logits
        dis_logits_in.append(-np.min(np.array(diss))) 
        dis_preds.append(np.argmin(np.array(diss)))

    return dis_logits_in, dis_logits_out, dis_preds



def KNN_classifier(testing_features, testing_labels, sorted_training_features):

    print("Begin KNN Classifier!")
    testing_similarity_logits = KNN_logits(testing_features, sorted_training_features)
    prediction_logits, predictions = np.amax(testing_similarity_logits, axis=1), np.argmax(testing_similarity_logits, axis=1)

    acc = accuracy_plain(predictions, testing_labels)
    print("KNN Accuracy is: ", acc)

    return prediction_logits, predictions, acc



def distance_classifier(testing_features, testing_labels, sorted_training_features):

    stats = feature_stats(sorted_training_features)
    dis_logits_in, dis_logits_out, dis_preds =  distances(stats, testing_features)

    acc = accuracy_plain(dis_preds, testing_labels)
    print("KNN Accuracy is: ", acc)

    return dis_logits_in, dis_logits_out, dis_preds, acc



def mahalanobis_stats(sorted_last_features, stats, ratio_in, ratio_out):
    
    cut_thresholds = []
    for c, (last_features_c, stats_c) in enumerate(zip(sorted_last_features, stats)):
        mu, var = stats_c
        m_distances = mahalanobis_group(last_features_c, mu, var)        
        m_distances = np.sort(m_distances)
        #print("m_distances", np.mean(m_distances))
        cut_index = int((1-ratio_out)*len(m_distances))   # outside
        cut_threshold = m_distances[cut_index]
        cut_thresholds.append((cut_threshold))

    return cut_thresholds


def feature_classifier(opt):

    with open(opt.exemplar_features_path, "rb") as f:
        features_exemplar_backbone, _, _, labels_examplar = pickle.load(f)         #
        #_, features_exemplar_backbone, _, labels_examplar = pickle.load(f)         #
        features_exemplar_backbone = np.squeeze(np.array(features_exemplar_backbone))

    sorted_exemplar_features = sortFeatures(features_exemplar_backbone, labels_examplar, opt)
    stats = feature_stats(sorted_exemplar_features)
    cut_thresholds = mahalanobis_stats(sorted_exemplar_features, stats, opt.downsample_ratio_center, opt.downsample_ratio)
    

    if opt.exemplar_features_path1 is not None:
        with open(opt.exemplar_features_path1, "rb") as f:       # !!!!!!!!
            features_exemplar_backbone1, _, _, labels_examplar1 = pickle.load(f)         #
            #_, features_exemplar_backbone, _, labels_examplar = pickle.load(f)         #
            features_exemplar_backbone1 = np.squeeze(np.array(features_exemplar_backbone1))
        features_exemplar_backbone_step1 = np.concatenate((features_exemplar_backbone, features_exemplar_backbone1), axis=1)
    
    if opt.exemplar_features_path2 is not None:
        with open(opt.exemplar_features_path2, "rb") as f:       # !!!!!!!!
            features_exemplar_backbone2, _, _, labels_examplar2 = pickle.load(f)         #
            #_, features_exemplar_backbone, _, labels_examplar = pickle.load(f)         #
            features_exemplar_backbone2 = np.squeeze(np.array(features_exemplar_backbone2))
        features_exemplar_backbone_step2 = np.concatenate((features_exemplar_backbone1, features_exemplar_backbone2), axis=1)

    sorted_features_examplar_backbone_step1 = sortFeatures(features_exemplar_backbone, labels_examplar, opt)
    sorted_features_examplar_backbone_step2 = sortFeatures(features_exemplar_backbone_step2, labels_examplar, opt)

    if  opt.testing_known_features_path is not None:
        with open(opt.testing_known_features_path, "rb") as f:
            features_testing_known_backbone, _, _, labels_testing_known = pickle.load(f)           #
            #_, features_testing_known_backbone, _, labels_testing_known = pickle.load(f)           #
            features_testing_known_backbone = np.squeeze(np.array(features_testing_known_backbone))
            labels_testing_known = np.squeeze(np.array(labels_testing_known))
    
    step1_indices = []
    for i, (testing_feature, testing_label) in enumerate(zip(features_testing_known_backbone, labels_testing_known)):
        mu, var = stats[testing_label]
        m_dis = mahalanobis(testing_feature, mu, np.linalg.inv(var))
        if m_dis < cut_thresholds[testing_label]:
            step1_indices.append(i)
    features_testing_known_backbone_step1 = [features_testing_known_backbone[i] for i in step1_indices]
    labels_testing_known_step1 = [labels_testing_known[i] for i in step1_indices]

    if opt.testing_known_features_path1 is not None:
        with open(opt.testing_known_features_path1, "rb") as f:             # !!!!!!!!
            features_testing_known_backbone1, _, _, labels_testing_known1 = pickle.load(f)           #
            #_, features_testing_known_backbone, _, labels_testing_known = pickle.load(f)           #
            features_testing_known_backbone1 = np.squeeze(np.array(features_testing_known_backbone1))
            labels_testing_known1 = np.squeeze(np.array(labels_testing_known1))
        features_testing_known_backbone = np.concatenate((features_testing_known_backbone, features_testing_known_backbone1), axis=1)

    features_testing_known_backbone_step2 = [features_testing_known_backbone[i] for i in range(len(features_testing_known_backbone)) if i not in step1_indices]
    labels_testing_known_step2 = [labels_testing_known1[i] for i in range(len(labels_testing_known1)) if i not in step1_indices]

    if opt.testing_known_features_path2 is not None:
        with open(opt.testing_known_features_path2, "rb") as f:             # !!!!!!!!
            features_testing_known_backbone2, _, _, labels_testing_known2 = pickle.load(f)           #
            #_, features_testing_known_backbone, _, labels_testing_known = pickle.load(f)           #
            features_testing_known_backbone2 = np.squeeze(np.array(features_testing_known_backbone2))
            labels_testing_known2 = np.squeeze(np.array(labels_testing_known2))
        features_testing_known_backbone = np.concatenate((features_testing_known_backbone, features_testing_known_backbone2), axis=1)


    features_testing_known_backbone_step1 = np.squeeze(np.array(features_testing_known_backbone_step1))
    features_testing_known_backbone_step2 = np.squeeze(np.array(features_testing_known_backbone_step2))
    labels_testing_known_step1 = np.squeeze(np.array(labels_testing_known_step1))
    labels_testing_known_step2 = np.squeeze(np.array(labels_testing_known_step2))
    sorted_features_testing_known_step1 = sortFeatures(features_testing_known_backbone_step1, labels_testing_known_step1, opt)
    sorted_features_testing_known_step2 = sortFeatures(features_testing_known_backbone_step2, labels_testing_known_step2, opt)
    
    #features_exemplar_backbone_step1 = [features_testing_known_backbone[i] for i in step1_indices]
    #features_exemplar_backbone_step2 = [features_testing_known_backbone[i] for i in range(len(features_exemplar_backbone)) and i not in step1_indices]
    #sorted_features_testing_known_step1 = sortFeatures(features_testing_known_backbone_step1, labels_testing_known_step1, opt)
    #sorted_features_testing_known_step2 = sortFeatures(features_testing_known_backbone_step2, labels_testing_known_step2, opt)

    features_testing_known_backbone_step1, labels_testing_known_step1 = down_sampling(features_testing_known_backbone_step1, labels_testing_known_step1, opt.downsampling_ratio_known)
    prediction_logits_known_step1, predictions_known_step1, acc_known_step1 = KNN_classifier(features_testing_known_backbone_step1, labels_testing_known_step1, sorted_features_examplar_backbone_step1)
    prediction_logits_known_dis_in_step1, prediction_logits_known_dis_out_step1, predictions_known_dis_step1, acc_known_dis_step1 = distance_classifier(features_testing_known_backbone_step1, labels_testing_known_step1, sorted_features_examplar_backbone_step1)

    features_testing_known_backbone_step2, labels_testing_known_step2 = down_sampling(features_testing_known_backbone_step2, labels_testing_known_step2, opt.downsampling_ratio_known)
    prediction_logits_known_step2, predictions_known_step2, acc_known_step2 = KNN_classifier(features_testing_known_backbone_step2, labels_testing_known_step2, sorted_features_examplar_backbone_step2)
    prediction_logits_known_dis_in_step2, prediction_logits_known_dis_out_step2, predictions_known_dis_step2, acc_known_dis_step2 = distance_classifier(features_testing_known_backbone_step2, labels_testing_known_step2, sorted_features_examplar_backbone_step2)


    with open(opt.testing_unknown_features_path, "rb") as f:
        features_testing_unknown_backbone, _, _, labels_testing_unknown = pickle.load(f)            #
        #_, features_testing_unknown_backbone, _, labels_testing_unknown = pickle.load(f)            #
        features_testing_unknown_backbone = np.squeeze(np.array(features_testing_unknown_backbone))
        labels_testing_unknown = np.squeeze(np.array(labels_testing_unknown))
        print("features_testing_unknown_backbone", features_testing_unknown_backbone.shape)

    step1_indices_unknown = []
    for i, (testing_feature, testing_label) in enumerate(zip(features_testing_unknown_backbone, labels_testing_unknown)):
        m_dis_min = 0
        index_min = 0
        for i, (mu, var) in enumerate(stats):
            m_dis = mahalanobis(testing_feature, mu, np.linalg.inv(var))
            if m_dis > m_dis_min:
                m_dis_min = m_dis
                index_min = i
        if m_dis_min < cut_thresholds[index_min]:
            step1_indices_unknown.append(i)
    print("step1_indices_unknown", len(step1_indices_unknown), step1_indices_unknown)
    features_testing_unknown_backbone_step1 = [features_testing_unknown_backbone[i] for i in step1_indices_unknown]
    labels_testing_unknown_step1 = [labels_testing_unknown[i] for i in step1_indices_unknown]
    features_testing_unknown_backbone_step1 = np.squeeze(np.array(features_testing_unknown_backbone_step1))
    labels_testing_unknown_step1 = np.squeeze(np.array(labels_testing_unknown_step1))

    if opt.testing_unknown_features_path1 is not None:
        with open(opt.testing_unknown_features_path1, "rb") as f:               # !!!!!!!!
            features_testing_unknown_backbone1, _, _, labels_testing_unknown1 = pickle.load(f)            #
            #_, features_testing_unknown_backbone, _, labels_testing_unknown = pickle.load(f)            #
            features_testing_unknown_backbone1 = np.squeeze(np.array(features_testing_unknown_backbone1))
            labels_testing_unknown1 = np.squeeze(np.array(labels_testing_unknown1))
            print("features_testing_known_backbone1", features_testing_unknown_backbone1.shape)
        features_testing_unknown_backbone = np.concatenate((features_testing_unknown_backbone, features_testing_unknown_backbone1), axis=1)

    features_testing_unknown_backbone_step2 = [features_testing_unknown_backbone[i] for i in range(len(features_testing_unknown_backbone)) if i not in step1_indices_unknown]
    labels_testing_unknown_step2 = [labels_testing_unknown1[i] for i in range(len(labels_testing_unknown1)) if i not in step1_indices_unknown]
    features_testing_unknown_backbone_step2 = np.squeeze(np.array(features_testing_unknown_backbone_step2))
    labels_testing_unknown_step2 = np.squeeze(np.array(labels_testing_unknown_step2))

    
    if opt.testing_unknown_features_path2 is not None:
        with open(opt.testing_unknown_features_path2, "rb") as f:               # !!!!!!!!
            features_testing_unknown_backbone2, _, _, labels_testing_unknown2 = pickle.load(f)            #
            #_, features_testing_unknown_backbone, _, labels_testing_unknown = pickle.load(f)            #
            features_testing_unknown_backbone2 = np.squeeze(np.array(features_testing_unknown_backbone2))
            labels_testing_unknown2 = np.squeeze(np.array(labels_testing_unknown2))
            print("features_testing_known_backbone2", features_testing_unknown_backbone2.shape)
        features_testing_unknown_backbone = np.concatenate((features_testing_unknown_backbone, features_testing_unknown_backbone2), axis=1)

    if len(step1_indices_unknown) > 0:
        features_testing_unknown_backbone_step1, labels_testing_unknown_step1 = down_sampling(features_testing_unknown_backbone_step1, labels_testing_unknown_step1, opt.downsampling_ratio_unknown)
        prediction_logits_unknown_step1, predictions_unknown_step1, _ = KNN_classifier(features_testing_unknown_backbone_step1, labels_testing_unknown_step1, sorted_features_examplar_backbone_step1)
        prediction_logits_unknown_dis_in_step1, prediction_logits_unknown_dis_out_step1, predictions_unknown_dis_step1, acc_unknown_dis_step1 = distance_classifier(features_testing_unknown_backbone_step1, labels_testing_unknown_step1, sorted_features_examplar_backbone_step1)
    else:
        prediction_logits_unknown_step1 = np.empty([1, 1])
        prediction_logits_unknown_dis_in_step1 = np.empty([1, 1])
        prediction_logits_unknown_dis_out_step1 = np.empty([1, 1])

    features_testing_unknown_backbone_step2, labels_testing_unknown_step2 = down_sampling(features_testing_unknown_backbone_step2, labels_testing_unknown_step2, opt.downsampling_ratio_unknown)
    prediction_logits_unknown_step2, predictions_unknown_step2, _ = KNN_classifier(features_testing_unknown_backbone_step2, labels_testing_unknown_step2, sorted_features_examplar_backbone_step2)
    prediction_logits_unknown_dis_in_step2, prediction_logits_unknown_dis_out_step2, predictions_unknown_dis_step2, acc_unknown_dis_step2 = distance_classifier(features_testing_unknown_backbone_step2, labels_testing_unknown_step2, sorted_features_examplar_backbone_step2)
    
    # Process results AUROC and OSCR
    # for AUROC, convert labels to binary labels, assume inliers are positive
    labels_binary_known_step1 = [1 if i < 100 else 0 for i in labels_testing_known_step1]
    labels_binary_known_step2 = [1 if i < 100 else 0 for i in labels_testing_known_step2]
    labels_binary_unknown_step1 = [1 if i < 100 else 0 for i in labels_testing_unknown_step1]
    labels_binary_unknown_step2 = [1 if i < 100 else 0 for i in labels_testing_unknown_step2]
    labels_binary_step1 = np.array(labels_binary_known_step1 + labels_binary_unknown_step1)
    labels_binary_step2 = np.array(labels_binary_known_step2 + labels_binary_unknown_step2)
    labels_binary = np.concatenate((labels_binary_step1, labels_binary_step2), axis=0)

    probs_binary_step2 = np.concatenate((prediction_logits_known_step2, prediction_logits_unknown_step2), axis=0) 
    if len(step1_indices_unknown) > 0:
        probs_binary_step1 = np.concatenate((prediction_logits_known_step1, prediction_logits_unknown_step1), axis=0)
    else:
        probs_binary_step1 =  prediction_logits_known_step1
    #print("probs_binary", probs_binary)
    probs_binary = np.concatenate((probs_binary_step1, probs_binary_step2), axis=0)

    """
    auroc_step1 = AUROC(labels_binary_step1, probs_binary_step1, opt)
    print("AUROC Step1 is: ", auroc_step1)
    auroc_step2 = AUROC(labels_binary_step2, probs_binary_step2, opt)
    print("AUROC Step2 is: ", auroc_step2)
    """
    auroc = AUROC(labels_binary, probs_binary, opt)
    print("AUROC Step is: ", auroc)

    probs_binary_dis_step2 = np.concatenate((prediction_logits_known_dis_in_step2, prediction_logits_unknown_dis_in_step2), axis=0)
    if len(step1_indices_unknown) > 0:
        probs_binary_dis_step1 = np.concatenate((prediction_logits_known_dis_in_step1, prediction_logits_unknown_dis_in_step1), axis=0)
    else:
        probs_binary_dis_step1 = prediction_logits_known_dis_in_step1
    #print("probs_binary", probs_binary_dis)
    probs_binary_dis = np.concatenate((probs_binary_dis_step1, probs_binary_dis_step2), axis=0)
    
    """
    auroc_step1 = AUROC(labels_binary_step1, probs_binary_dis_step1, opt)
    print("Dis AUROC Step1 is: ", auroc_step1)
    auroc_step2 = AUROC(labels_binary_step2, probs_binary_dis_step2, opt)
    print("Dis AUROC Step2 is: ", auroc_step2)
    """

    auroc = AUROC(labels_binary, probs_binary_dis, opt)
    print("Dis AUROC is: ", auroc)

    prediction_logits_known_dis_out = np.array(prediction_logits_known_dis_out_step1 + prediction_logits_known_dis_out_step2)
    predictions_known = np.concatenate((predictions_known_step1, predictions_known_step2), axis=0)
    predictions_known_dis = np.array(predictions_known_dis_step1+predictions_known_dis_step2)
    labels_testing_known = np.concatenate((labels_testing_known_step1, labels_testing_known_step2), axis=0)
    if len(step1_indices_unknown) > 0:
        prediction_logits_unknown_dis_out = np.array(prediction_logits_unknown_dis_out_step1 + prediction_logits_unknown_dis_out_step2)
        predictions_unknown = np.concatenate((predictions_unknown_step1, predictions_unknown_step2), axis=0)
        predictions_unknown_dis = np.array(predictions_unknown_dis_step1 + predictions_unknown_dis_step2)
        labels_testing_unknown = np.concatenate((labels_testing_unknown_step1, labels_testing_unknown_step2), axis=0)
    else:
        prediction_logits_unknown_dis_out = np.array(prediction_logits_unknown_dis_out_step2)
        predictions_unknown = predictions_unknown_step2
        predictions_unknown_dis = np.array(predictions_unknown_dis_step2)
        labels_testing_unknown =  labels_testing_unknown_step2

    predictions = predictions_known
    predictions_dis = predictions_known_dis
    labels_testing = labels_testing_known

    print("KNN Accuracy", accuracy_plain(predictions, labels_testing))
    print("DIstance Accuracy", accuracy_plain(predictions_dis, labels_testing))

    # OSCR
    oscr = OSCR(prediction_logits_known_dis_out, prediction_logits_unknown_dis_out, predictions_known, labels_testing_known)
    print("OSCR is: ", oscr)
    #print("Acc Known: ", acc_known)

    return auroc             # oscr, acc_known

        

if __name__ == "__main__":
    
    opt = parse_option()

    models, linear_model = set_model(opt)
    print("Model loaded!!")
    
    auroc = feature_classifier(opt)                        # oscr, acc_known

    #data_loader = set_loader(opt)
    #avg_accuracy = testing_nn_classifier(models, linear_model, data_loader)
    #print("ID", opt.trail, "Average NN accuracy on inlier testing data is: ", avg_accuracy)
  