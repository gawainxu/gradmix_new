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

from networks.resnet_big import SupConResNet, LinearClassifier
from networks.resnet_preact import SupConpPreactResNet
from networks.simCNN import simCNN_contrastive

from util import  feature_stats
from util import accuracy, AverageMeter, accuracy_plain, AUROC, OSCR, down_sampling
from distance_utils  import sortFeatures
from datautil import get_test_datasets, get_outlier_datasets

from sklearn.neighbors import LocalOutlierFactor


torch.multiprocessing.set_sharing_strategy('file_system')

def parse_option():

    parser = argparse.ArgumentParser('argument for feature reading')

    parser.add_argument('--datasets', type=str, default='tinyimgnet',
                        choices=["cifar-10-100-10", "cifar-10-100-50", 'cifar10', 'cifar100', "tinyimgnet", 'mnist', "svhn", "imagenet100"], help='dataset')
    parser.add_argument('--data_folder', type=str, default=None, help='path to custom dataset')
    parser.add_argument('--model', type=str, default="resnet18",  choices=["resnet18", "resnet34", "preactresnet18", "preactresnet34", "simCNN"])
    parser.add_argument("--model_path", type=str, default="/save/SupCon/cifar10_models/cifar10_resnet18_original_data__vanilia__SimCLR_0.01_trail_0/ckpt_epoch_400.pth")
    parser.add_argument("--model_path1", type=str, default=None)
    parser.add_argument("--model_path2", type=str, default=None)
    parser.add_argument("--end", type=bool, default=False, help="if it is end to end training")
    parser.add_argument("--ensembles", type=int, default=1)
    parser.add_argument("--linear_model_path", type=str, default="/save/SupCon/cifar10_models/cifar10_resnet18_original_data__vanilia__SimCLR_0.01_trail_0/last_linear.pth")
    parser.add_argument("--num_classes", type=int, default=100)
    parser.add_argument("--feat_dim", type=int, default=128)

    parser.add_argument("--exemplar_features_path", type=str,
                        default="/features/imagenet100_vgg16_original_data__vanilia__Joint_0.4_0.6_trail_0_128_256_split_128_600_outliers_0_train")
    parser.add_argument("--testing_known_features_path", type=str,
                        default="/features/imagenet100_vgg16_original_data__vanilia__Joint_0.4_0.6_trail_0_128_256_split_128_600_outliers_0_test_known")
    parser.add_argument("--testing_unknown_features_path", type=str,
                        default="/features/imagenet100_vgg16_original_data__vanilia__Joint_0.4_0.6_trail_0_128_256_split_128_600_outliers_0_test_unknown")

    parser.add_argument("--exemplar_features_path1", type=str,
                        default=None)
    parser.add_argument("--testing_known_features_path1", type=str,
                        default=None)
    parser.add_argument("--testing_unknown_features_path1", type=str,
                        default=None)

    parser.add_argument("--exemplar_features_path2", type=str,
                        default=None)
    parser.add_argument("--testing_known_features_path2", type=str,
                        default=None)
    parser.add_argument("--testing_unknown_features_path2", type=str,
                        default=None)

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
    parser.add_argument('--num_workers', type=int, default=4, help='num of workers to use')
    parser.add_argument("--downsampling_ratio_known", type=int, default=10)
    parser.add_argument("--downsampling_ratio_unknown", type=int, default=10)
    parser.add_argument("--ensemble_features", type=bool, default=False)

    parser.add_argument("--K", type=int, default=3)
    parser.add_argument("--LoF_K", type=int, default=5)
    parser.add_argument("--LoF_contamination", type=float, default=0.01)

    parser.add_argument("--auroc_save_path", type=str, default="/plots/auroc")

    parser.add_argument("--with_outliers", type=bool, default=False)
    parser.add_argument("--downsample", type=bool, default=False)
    parser.add_argument("--last_feature_path", type=str, default=None)
    parser.add_argument("--downsample_ratio", type=float, default=0)

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
    opt.prediction_save_path = opt.exemplar_features_path.split("/")[-1]
    opt.prediction_save_path.replace("_train", "")
    opt.prediction_save_path = opt.prediction_save_path + "_predictions"
    opt.prediction_save_path = opt.main_dir + "/" + opt.prediction_save_path

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


def load_model(model, linear_model=None, path=None):
    ckpt = torch.load(path, map_location='cpu')
    state_dict = ckpt['model']

    new_state_dict = {}
    for k, v in state_dict.items():
        k = k.replace("module.", "")
        new_state_dict[k] = v
    state_dict = new_state_dict
    model.load_state_dict(state_dict)

    if linear_model is not None:
        state_dict_linear = ckpt['linear']
        new_state_dict = {}
        for k, v in state_dict_linear.items():
            k = k.replace("module.", "")
            new_state_dict[k] = v
        state_dict = new_state_dict
        linear_model.load_state_dict(state_dict)

        return model, linear_model
    
    return model


def set_model(opt):

    if opt.datasets == "mnist":
        in_channels = 1
    elif opt.datasets == "FUB":
        in_channels = 1
    else:
        in_channels = 3

    opt.num_ensembles = 1

    if opt.model == "resnet18" or opt.model == "resnet34":
        model = SupConResNet(name=opt.model, feat_dim=opt.feat_dim, in_channels=in_channels)
    elif opt.model == "preactresnet18" or opt.model == "preactresnet34":
        model = SupConpPreactResNet(name=opt.model, feat_dim=opt.feat_dim, in_channels=in_channels)
    else:
        model = simCNN_contrastive(opt)

    model.eval()
    model = model.cpu()
    models = []
    models.append(model)

    if opt.model_path1 is not None:
        model1 = copy.deepcopy(model)
        model1 = load_model(model1, opt.model_path1)
        models.append(model1)
        opt.num_ensembles += 1

    if opt.model_path2 is not None:
        model2 = copy.deepcopy(model)
        model2 = load_model(model2, opt.model_path2)
        models.append(model2)
        opt.num_ensembles += 1

    return models



def set_loader(opt):
    # construct data loader
    test_dataset = get_test_datasets(opt)
    outlier_dataset = get_outlier_datasets(opt)
    
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=opt.batch_size, shuffle=True,
                                              num_workers=opt.num_workers, pin_memory=True)
    outlier_loader = torch.utils.data.DataLoader(outlier_dataset, batch_size=opt.batch_size, shuffle=True,
                                              num_workers=opt.num_workers, pin_memory=True)

    return test_loader, outlier_loader


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



def LoF(testing_features, sorted_exemplar_features, opt):
    
    scores = []
    for idx, testing_feature in enumerate(testing_features):
        
        clost_points = None
        clost_similarities = 0
        
        for training_features_c in sorted_exemplar_features:
            
            training_features_c = np.array(training_features_c, dtype=float)
            similarities = np.matmul(training_features_c, testing_feature) / np.linalg.norm(training_features_c, axis=1) / np.linalg.norm(testing_feature)
            if np.sum(similarities) > clost_similarities:
                clost_points = training_features_c
                clost_similarities = np.sum(similarities)
                
        # start LoF
        clf = LocalOutlierFactor(n_neighbors=opt.LoF_K, novelty=True, metric="cosine", contamination=opt.LoF_contamination)
        clf.fit(clost_points)
        testing_feature = np.expand_dims(testing_feature, axis=0)
        score = clf.decision_function(testing_feature)
        scores.append(score)
        
    return scores


def distances(stats, test_features, mode="others"):

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
        features_exemplar_head, features_exemplar_backbone, _, labels_examplar = pickle.load(f)
        features_exemplar_head = np.squeeze(np.array(features_exemplar_head))    
        features_exemplar_backbone = np.squeeze(np.array(features_exemplar_backbone))
        print("features_exemplar_head", features_exemplar_head.shape)

    if opt.exemplar_features_path1 is not None:
        with open(opt.exemplar_features_path1, "rb") as f:       
            features_exemplar_head1, features_exemplar_backbone1, _, labels_examplar1 = pickle.load(f)         
            features_exemplar_head1 = np.squeeze(np.array(features_exemplar_head1))
            features_exemplar_backbone1 = np.squeeze(np.array(features_exemplar_backbone1))
        features_exemplar_head = np.concatenate((features_exemplar_head, features_exemplar_head1), axis=1)
        features_exemplar_backbone = np.concatenate((features_exemplar_backbone, features_exemplar_backbone1), axis=1)
    
    if opt.exemplar_features_path2 is not None:
        with open(opt.exemplar_features_path2, "rb") as f:       
            features_exemplar_head2, features_exemplar_backbone2, _, labels_examplar2 = pickle.load(f)         
            features_exemplar_head2 = np.squeeze(np.array(features_exemplar_head2))
            features_exemplar_backbone2 = np.squeeze(np.array(features_exemplar_backbone2))
        features_exemplar_head = np.concatenate((features_exemplar_head, features_exemplar_head2), axis=1)
        features_exemplar_backbone = np.concatenate((features_exemplar_backbone, features_exemplar_backbone2), axis=1)


    if opt.ensemble_features is True:
        features_exemplar_head = np.concatenate((features_exemplar_backbone, features_exemplar_head), axis=1)
    sorted_features_examplar_head = sortFeatures(features_exemplar_head, labels_examplar, opt)
    #sorted_features_examplar_backbone = sortFeatures(features_exemplar_backbone, labels_examplar, opt)

    if opt.testing_known_features_path is not None:
        with open(opt.testing_known_features_path, "rb") as f:
            features_testing_known_head, features_testing_known_backbone, _, labels_testing_known = pickle.load(f)
            features_testing_known_head = np.squeeze(np.array(features_testing_known_head))       
            features_testing_known_backbone = np.squeeze(np.array(features_testing_known_backbone))
            labels_testing_known = np.squeeze(np.array(labels_testing_known))
    
    if opt.testing_known_features_path1 is not None:
        with open(opt.testing_known_features_path1, "rb") as f:             
            features_testing_known_head1, features_testing_known_backbone1, _, labels_testing_known1 = pickle.load(f)          
            features_testing_known_head1 = np.squeeze(np.array(features_testing_known_head1))       
            features_testing_known_backbone1 = np.squeeze(np.array(features_testing_known_backbone1))
            labels_testing_known1 = np.squeeze(np.array(labels_testing_known1))
        features_testing_known_head = np.concatenate((features_testing_known_head, features_testing_known_head1), axis=1)
        features_testing_known_backbone = np.concatenate((features_testing_known_backbone, features_testing_known_backbone1), axis=1)

    if opt.testing_known_features_path2 is not None:
        with open(opt.testing_known_features_path2, "rb") as f:             
            features_testing_known_head2, features_testing_known_backbone2, _, labels_testing_known2 = pickle.load(f)           
            features_testing_known_head2 = np.squeeze(np.array(features_testing_known_head2))
            features_testing_known_backbone2 = np.squeeze(np.array(features_testing_known_backbone2))
            labels_testing_known2 = np.squeeze(np.array(labels_testing_known2))
        features_testing_known_head = np.concatenate((features_testing_known_head, features_testing_known_head2), axis=1)
        features_testing_known_backbone = np.concatenate((features_testing_known_backbone, features_testing_known_backbone2), axis=1)

    if opt.ensemble_features is True:
        features_testing_known_head = np.concatenate((features_testing_known_backbone, features_testing_known_head), axis=1)
    #features_testing_known_head, labels_testing_known = down_sampling(features_testing_known_head, labels_testing_known, opt.downsampling_ratio_known)
    #features_testing_known_backbone, labels_testing_known = down_sampling(features_testing_known_backbone, labels_testing_known, opt.downsampling_ratio_known)
    #prediction_logits_known, predictions_known, acc_known = KNN_classifier(features_testing_known_backbone, labels_testing_known, sorted_features_examplar_backbone)
    prediction_logits_known, predictions_known, acc_known = KNN_classifier(features_testing_known_head, labels_testing_known, sorted_features_examplar_head)

    prediction_logits_known_dis_in, prediction_logits_known_dis_out, predictions_known_dis, acc_known_dis = distance_classifier(features_testing_known_head, labels_testing_known, sorted_features_examplar_head)

    with open(opt.testing_unknown_features_path, "rb") as f:
        features_testing_unknown_head, features_testing_unknown_backbone, _, labels_testing_unknown = pickle.load(f)          
        features_testing_unknown_head = np.squeeze(np.array(features_testing_unknown_head))
        features_testing_unknown_backbone = np.squeeze(np.array(features_testing_unknown_backbone))
        labels_testing_unknown = np.squeeze(np.array(labels_testing_unknown))

    if opt.testing_unknown_features_path1 is not None:
        with open(opt.testing_unknown_features_path1, "rb") as f:            
            features_testing_unknown_head1, features_testing_unknown_backbone1, _, labels_testing_unknown1 = pickle.load(f)            
            features_testing_unknown_head1 = np.squeeze(np.array(features_testing_unknown_head1))
            features_testing_unknown_backbone1 = np.squeeze(np.array(features_testing_unknown_backbone1))
            labels_testing_unknown1 = np.squeeze(np.array(labels_testing_unknown1))
        features_testing_unknown_head = np.concatenate((features_testing_unknown_head, features_testing_unknown_head1), axis=1)
        features_testing_unknown_backbone = np.concatenate((features_testing_unknown_backbone, features_testing_unknown_backbone1), axis=1)
    
    if opt.testing_unknown_features_path2 is not None:
        with open(opt.testing_unknown_features_path2, "rb") as f:               
            features_testing_unknown_head2, features_testing_unknown_backbone2, _, labels_testing_unknown2 = pickle.load(f)            
            features_testing_unknown_head2 = np.squeeze(np.array(features_testing_unknown_head2))
            features_testing_unknown_backbone2 = np.squeeze(np.array(features_testing_unknown_backbone2))
            labels_testing_unknown2 = np.squeeze(np.array(labels_testing_unknown2))
        features_testing_unknown_head = np.concatenate((features_testing_unknown_head, features_testing_unknown_head2), axis=1)
        features_testing_unknown_backbone = np.concatenate((features_testing_unknown_backbone, features_testing_unknown_backbone2), axis=1)

    if opt.ensemble_features is True:
        features_testing_unknown_head = np.concatenate((features_testing_unknown_backbone, features_testing_unknown_head), axis=1)
    #, labels_testing_unknown = down_sampling(features_testing_unknown_head, labels_testing_unknown, opt.downsampling_ratio_unknown)
    #features_testing_unknown_backbone, labels_testing_unknown = down_sampling(features_testing_unknown_backbone, labels_testing_unknown, opt.downsampling_ratio_unknown)
    #prediction_logits_unknown, predictions_unknown, _ = KNN_classifier(features_testing_unknown_backbone, labels_testing_unknown, sorted_features_examplar_backbone)
    prediction_logits_unknown, predictions_unknown, _ = KNN_classifier(features_testing_unknown_head, labels_testing_unknown, sorted_features_examplar_head)    #
     
    prediction_logits_unknown_dis_in, prediction_logits_unknown_dis_out, predictions_unknown_dis, acc_unknown_dis = distance_classifier(features_testing_unknown_head, labels_testing_unknown, sorted_features_examplar_head)
    
    knn_predictions = np.concatenate((predictions_known, predictions_unknown), axis=0)
    distance_predictions = np.concatenate((predictions_known_dis, predictions_unknown_dis), axis=0)
    labels_testing = np.concatenate((labels_testing_known, labels_testing_unknown), axis=0)

    with open(opt.prediction_save_path, "wb") as f:
        pickle.dump((knn_predictions, distance_predictions, labels_testing), f)

    # Process results AUROC and OSCR
    # for AUROC, convert labels to binary labels, assume inliers are positive
    labels_binary_known = [1 if i < opt.num_classes else 0 for i in labels_testing_known]
    labels_binary_unknown = [1 if i < opt.num_classes else 0 for i in labels_testing_unknown]
    labels_binary = np.array(labels_binary_known + labels_binary_unknown)
    #print("labels_binary", labels_binary)

    probs_binary = np.concatenate((prediction_logits_known, prediction_logits_unknown), axis=0) 

    auroc = AUROC(labels_binary, probs_binary, opt)
    print("AUROC is: ", auroc)

    probs_binary_dis = np.concatenate((prediction_logits_known_dis_in, prediction_logits_unknown_dis_in), axis=0) 
    #print("probs_binary", probs_binary_dis)

    auroc = AUROC(labels_binary, probs_binary_dis, opt)
    print("Dis AUROC is: ", auroc)
    
    # AUROC based on LoF
    #all_testing_features = np.concatenate((features_testing_known_backbone, features_testing_unknown_backbone), axis=0)
    #scores = LoF(all_testing_features, sorted_features_examplar_backbone, opt)
    #auroc = AUROC(labels_binary, scores, opt)
    #print("LoF AUROC is: ", auroc)

    # OSCR
    #oscr = OSCR(np.array(prediction_logits_known_dis_out), np.array(prediction_logits_unknown_dis_out), predictions_known, labels_testing_known)
    #print("OSCR is: ", oscr)

    #print("Acc Known: ", acc_known)

    return 0             # oscr, acc_known

        
if __name__ == "__main__":
    
    opt = parse_option()

    #models, linear_model = set_model(opt)
    #print("Model loaded!!")
    
    auroc = feature_classifier(opt)                        # oscr, acc_known