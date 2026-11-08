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
import matplotlib.pyplot as plt

import torch
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from torchvision import datasets
import numpy as np
import pickle
import copy
from itertools import chain
from scipy.spatial.distance import mahalanobis

from util import  feature_stats
from util import accuracy, AverageMeter, accuracy_plain, AUROC, OSCR, down_sampling
from distance_utils  import sortFeatures
from datautil import get_test_datasets, get_outlier_datasets
from datautil import num_inlier_classes_mapping, downsample_data

from sklearn.neighbors import LocalOutlierFactor


torch.multiprocessing.set_sharing_strategy('file_system')

def parse_option():

    parser = argparse.ArgumentParser('argument for feature reading')

    parser.add_argument('--datasets', type=str, default='cifar10',
                        choices=["cifar-10-100-10", "cifar-10-100-50", 'cifar10', "tinyimgnet", 'mnist', "svhn"], help='dataset')
    parser.add_argument('--data_folder', type=str, default=None, help='path to custom dataset')
    parser.add_argument("--resnet_wide", type=int, default=1, help="factor for expanding channels in wide resnet")
    parser.add_argument("--num_classes", type=int, default=6)
    parser.add_argument("--feat_dim", type=int, default=128)
    
    parser.add_argument("--exemplar_features_path", type=str, default="/features/cifar10_resnet18_1trail_0_128_256_randaug2_6_100_train")
    parser.add_argument("--testing_known_features_path", type=str, default="/features/cifar10_resnet18_1trail_0_128_256_randaug2_6_100_test_known")
    parser.add_argument("--testing_unknown_features_path", type=str, default="/features/cifar10_resnet18_1trail_0_128_256_randaug2_6_100_test_unknown")

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
    parser.add_argument("--if_train", type=str, default="test_known", choices=['train', 'val', 'test_known', 'test_unknown'])
    parser.add_argument('--batch_size', type=int, default=1, help='batch_size')

    parser.add_argument("--K", type=int, default=3)
    parser.add_argument("--LoF_K", type=int, default=5)
    parser.add_argument("--LoF_contamination", type=float, default=0.01)

    parser.add_argument("--auroc_save_path", type=str, default="/plots/auroc")

    parser.add_argument("--with_outliers", type=bool, default=False)
    parser.add_argument("--downsample", type=bool, default=False)
    parser.add_argument("--last_feature_path", type=str, default=None)
    parser.add_argument("--downsample_ratio_train", type=float, default=None)
    parser.add_argument("--downsampling_ratio_known", type=int, default=10)
    parser.add_argument("--downsampling_ratio_unknown", type=int, default=10)

    opt = parser.parse_args()
    opt.main_dir = os.getcwd()

    opt.exemplar_features_path = opt.main_dir + opt.exemplar_features_path
    opt.testing_known_features_path = opt.main_dir + opt.testing_known_features_path
    opt.testing_unknown_features_path = opt.main_dir + opt.testing_unknown_features_path
    opt.auroc_save_path = opt.main_dir + opt.auroc_save_path
    opt.prediction_save_path = opt.exemplar_features_path.split("/")[-1]
    opt.prediction_save_path.replace("_train", "")
    opt.prediction_save_path = opt.prediction_save_path + "_predictions"
    opt.prediction_save_path = opt.main_dir + "/" + opt.prediction_save_path


    return opt



def testing_nn_classifier(models, classifier, dataloader):

    for model in models:
        model.eval()
    classifier.eval()

    top1 = AverageMeter()
    scores_max = []
    preds = []
    labels = []

    for idx, (images, label) in enumerate(dataloader):

        #print(idx)
        #images = images.cuda(non_blocking=True)
        #labels = labels.cuda(non_blocking=True)
        bsz = label.shape[0]

        features = torch.empty((bsz, 0), dtype=torch.float32)
        for model in models:
            feature = model.encoder(images)
            features = torch.cat((features, feature), dim=1)
        output = classifier(features)

        acc, pred, score_max = accuracy(output, label)
        top1.update(acc, bsz)
        scores_max.append(score_max.numpy())
        preds.append(pred)
        labels.append(label)

    return top1.avg, scores_max, preds, labels


def KNN_logits(testing_features, sorted_exemplar_features):

    testing_similarity_logits = []

    #testing_features = testing_features.astype(np.double)
    #testing_features = testing_features / np.linalg.norm(testing_features, axis=1)[:, np.newaxis]  ####

    for idx, testing_feature in enumerate(testing_features):
        #print(idx)
        similarity_logits = []
        for training_features_c in sorted_exemplar_features:
            
            training_features_c = np.array(training_features_c, dtype=float)
            #training_features_c = training_features_c[::2]                                                          # TODO
            
            similarities = np.matmul(training_features_c, testing_feature) / np.linalg.norm(training_features_c, axis=1) / np.linalg.norm(testing_feature)
            ind = np.argsort(similarities)[-opt.K:]
            top_k_similarities = similarities[ind]
            similarity_logits.append(np.sum(top_k_similarities))      #!!!!
            #similarity_logits.append(top_k_similarities[-1])
            
            """
            training_features_c = training_features_c.astype(np.double)
            training_features_c = training_features_c / np.linalg.norm(training_features_c, axis=1)[:, np.newaxis]
            diff = training_features_c - testing_feature
            diff = diff.astype(np.double)
            similarities = np.linalg.norm((diff), axis=1)
            similarity_logits.append(np.min(similarities)) 
            """

        testing_similarity_logits.append(similarity_logits)
    
    testing_similarity_logits = np.array(testing_similarity_logits)
    testing_similarity_logits = np.divide(testing_similarity_logits.T, np.sum(testing_similarity_logits, axis=1)).T                         # normalization, maybe not necessary???

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
    #prediction_logits, predictions = -np.amin(testing_similarity_logits, axis=1), np.argmin(testing_similarity_logits, axis=1)       # minus here, larger score for inliers

    acc = accuracy_plain(predictions, testing_labels)
    print("KNN Accuracy is: ", acc)

    return prediction_logits, predictions, acc


def distance_classifier(testing_features, testing_labels, sorted_training_features):

    stats = feature_stats(sorted_training_features)
    dis_logits_in, dis_logits_out, dis_preds =  distances(stats, testing_features)

    acc = accuracy_plain(dis_preds, testing_labels)
    print("Distance Accuracy is: ", acc)

    return dis_logits_in, dis_logits_out, dis_preds, acc


def feature_classifier(opt):

    with open(opt.exemplar_features_path, "rb") as f:
        features_exemplar, labels_examplar = pickle.load(f) 
        features_exemplar = np.squeeze(np.array(features_exemplar))    

    sorted_features_examplar = sortFeatures(features_exemplar, labels_examplar, opt)


    if opt.downsample_ratio_train is not None:
        new_features_inlier = []
        for i, c_features in enumerate(sorted_features_examplar):
            print(i)
            c_features = np.array(c_features)
            downsampled_data = downsample_data(c_features, downsample_ratio=opt.downsample_ratio_train)
            new_features_inlier.append(downsampled_data)
        sorted_features_examplar = new_features_inlier


    if opt.testing_known_features_path is not None:
        with open(opt.testing_known_features_path, "rb") as f:
            features_testing_known, labels_testing_known = pickle.load(f) 
            features_testing_known = np.squeeze(np.array(features_testing_known))       
            labels_testing_known = np.squeeze(np.array(labels_testing_known))
    

    features_testing_known, labels_testing_known = down_sampling(features_testing_known, labels_testing_known, opt.downsampling_ratio_known)
    prediction_logits_known, predictions_known, acc_known = KNN_classifier(features_testing_known, labels_testing_known, sorted_features_examplar)
    #prediction_logits_known_dis_in, prediction_logits_known_dis_out, predictions_known_dis, acc_known_dis = distance_classifier(features_testing_known, labels_testing_known, sorted_features_examplar)


    with open(opt.testing_unknown_features_path, "rb") as f:
        features_testing_unknown, labels_testing_unknown = pickle.load(f)          
        features_testing_unknown = np.squeeze(np.array(features_testing_unknown))
        labels_testing_unknown = np.squeeze(np.array(labels_testing_unknown))
        

    features_testing_unknown, labels_testing_unknown = down_sampling(features_testing_unknown, labels_testing_unknown, opt.downsampling_ratio_unknown)
    prediction_logits_unknown, predictions_unknown, _ = KNN_classifier(features_testing_unknown, labels_testing_unknown, sorted_features_examplar)
    #prediction_logits_unknown_dis_in, prediction_logits_unknown_dis_out, predictions_unknown_dis, acc_unknown_dis = distance_classifier(features_testing_unknown, labels_testing_unknown, sorted_features_examplar)
    
    knn_predictions = np.concatenate((predictions_known, predictions_unknown), axis=0)
    #distance_predictions = np.concatenate((predictions_known_dis, predictions_unknown_dis), axis=0)
    labels_testing = np.concatenate((labels_testing_known, labels_testing_unknown), axis=0)

   
    # Process results AUROC and OSCR
    # for AUROC, convert labels to binary labels, assume inliers are positive
    labels_binary_known = [1 if i < 100 else 0 for i in labels_testing_known]
    labels_binary_unknown = [1 if i < 100 else 0 for i in labels_testing_unknown]
    labels_binary = np.array(labels_binary_known + labels_binary_unknown)

    probs_binary = np.concatenate((prediction_logits_known, prediction_logits_unknown), axis=0) 

    
    auroc = AUROC(labels_binary, probs_binary, opt)
    print("AUROC is: ", auroc)

    #probs_binary_dis = np.concatenate((prediction_logits_known_dis_in, prediction_logits_unknown_dis_in), axis=0) 

    #auroc = AUROC(labels_binary, probs_binary_dis, opt)
    #print("Dis AUROC is: ", auroc)
    

    return auroc             # oscr, acc_known

        
if __name__ == "__main__":
    
    opt = parse_option()

    #models, linear_model = set_model(opt)
    #print("Model loaded!!")
    
    auroc = feature_classifier(opt)                        # oscr, acc_known
    
    """
    models, linear_model = set_model(opt)
    test_loader, outlier_loader = set_loader(opt)
    avg_accuracy_test, scores_max_test, preds, labels = testing_nn_classifier(models, linear_model, test_loader)
    _, scores_max_outlier, _, _ = testing_nn_classifier(models, linear_model, outlier_loader)
    #with open("./scores", "wb") as f:
    #    pickle.dump((scores_max_test, scores_max_outlier), f)
    print("ID", opt.trail, "Average NN accuracy on inlier testing data is: ", avg_accuracy_test)

    labels_binary_known = [1 for _ in range(len(scores_max_test))]
    labels_binary_unknown = [0 for _ in range(len(scores_max_outlier))]
    labels_binary = np.array(labels_binary_known + labels_binary_unknown)
    scores_binary = np.array(scores_max_test + scores_max_outlier)
    auroc = AUROC(labels_binary, scores_binary, opt)
    print("NN AUROC is: ", auroc)

    scores_max_test = np.array(scores_max_test)
    scores_max_outlier = np.array(scores_max_outlier)
    oscr = OSCR(-scores_max_test, scores_max_outlier, preds, labels)
    print("NN OSCR is: ", oscr)
    """


    """
    1. use penultimate layer instead of head
    2. use ecudien distance 
    3. use kth distance instead of average distance
    4. feature normalization
    5. downsample training data
    """

    """
    pay attention to the samples at boundary
    """