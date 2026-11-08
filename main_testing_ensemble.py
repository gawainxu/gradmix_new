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

from networks.resnet_multi import SupConResNet_end, SupConResNet_inter

from util import feature_stats
from util import accuracy, AverageMeter, accuracy_plain, AUROC, OSCR, down_sampling
from distance_utils import sortFeatures
from datautil import get_test_datasets, get_outlier_datasets

from sklearn.neighbors import LocalOutlierFactor
from scipy import stats

torch.multiprocessing.set_sharing_strategy('file_system')


def parse_option():
    parser = argparse.ArgumentParser('argument for feature reading')

    parser.add_argument('--datasets', type=str, default='cifar10',
                        choices=["cifar-10-100-10", "cifar-10-100-50", 'cifar10', "tinyimgnet", 'mnist', "svhn"],
                        help='dataset')
    parser.add_argument('--data_folder', type=str, default=None, help='path to custom dataset')
    parser.add_argument('--model', type=str, default="resnet18",
                        choices=["resnet18", "resnet34", "preactresnet18", "preactresnet34", "simCNN"])
    parser.add_argument("--model_path", type=str,
                        default="/save/SupCon/cifar10_models/cifar10_resnet18_ensemble_trail_1_128_256_end_0.5_0.05_0.01_1.0_1.0_1.0/last.pth")
    parser.add_argument("--num_classes", type=int, default=6)
    parser.add_argument("--feat_dim", type=int, default=128)

    parser.add_argument("--exemplar_features_path", type=str, default="/features1/cifar10_resnet18_ensemble_trail_1_128_256_end_0.5_0.05_0.01_1.0_1.0_1.0_600_train")
    parser.add_argument("--testing_known_features_path", type=str, default="/features1/cifar10_resnet18_ensemble_trail_1_128_256_end_0.5_0.05_0.01_1.0_1.0_1.0_600_test_known")
    parser.add_argument("--testing_unknown_features_path", type=str, default="/features1/cifar10_resnet18_ensemble_trail_1_128_256_end_0.5_0.05_0.01_1.0_1.0_1.0_600_test_unknown")
    parser.add_argument("--ensemble_mode", type=str, default="end")

    parser.add_argument("--trail", type=int, default=1)
    parser.add_argument("--split_train_val", type=bool, default=True)
    parser.add_argument("--action", type=str, default="testing_known",
                        choices=["training_supcon", "trainging_linear", "testing_known", "testing_unknown",
                                 "feature_reading"])
    parser.add_argument('--method', type=str, default='SupCon',
                        choices=['SupCon', 'SimCLR'], help='choose method')
    parser.add_argument("--temp", type=str, default=0.5)
    parser.add_argument("--lr", type=str, default=0.001)
    parser.add_argument("--training_bz", type=int, default=200)
    parser.add_argument("--mem_size", type=int, default=500)
    parser.add_argument("--if_train", type=str, default="test_known",
                        choices=['train', 'val', 'test_known', 'test_unknown'])
    parser.add_argument('--batch_size', type=int, default=1, help='batch_size')
    parser.add_argument('--num_workers', type=int, default=4, help='num of workers to use')
    parser.add_argument("--downsampling_ratio_known", type=int, default=10)
    parser.add_argument("--downsampling_ratio_unknown", type=int, default=10)
    parser.add_argument("--ensemble_features", type=bool, default=False)

    parser.add_argument("--K", type=int, default=10)
    parser.add_argument("--LoF_K", type=int, default=5)
    parser.add_argument("--LoF_contamination", type=float, default=0.01)

    parser.add_argument("--auroc_save_path", type=str,
                        default="/plots/cifar10_resnet18_temp_0.005_id_4_lr_0.001_bz_256_auroc.pdf")

    parser.add_argument("--with_outliers", type=bool, default=False)
    parser.add_argument("--downsample", type=bool, default=False)
    parser.add_argument("--last_feature_path", type=str, default=None)
    parser.add_argument("--downsample_ratio", type=float, default=0)

    opt = parser.parse_args()
    opt.main_dir = os.getcwd()
    opt.model_path = opt.main_dir + opt.model_path

    opt.auroc_save_path = opt.main_dir + opt.auroc_save_path
    opt.prediction_save_path = opt.exemplar_features_path.split("/")[-1]
    opt.prediction_save_path.replace("_train", "")
    opt.prediction_save_path = opt.prediction_save_path + "_predictions"
    opt.prediction_save_path = opt.main_dir + "/" + opt.prediction_save_path


    if opt.exemplar_features_path is not None:
        opt.exemplar_features_path = opt.main_dir + opt.exemplar_features_path
    if opt.testing_known_features_path is not None:
        opt.testing_known_features_path = opt.main_dir + opt.testing_known_features_path
    if opt.testing_unknown_features_path is not None:
        opt.testing_unknown_features_path = opt.main_dir + opt.testing_unknown_features_path

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
    else:
        in_channels = 3

    if opt.ensemble_mode == "end":
        model = SupConResNet_end(name=opt.model, feat_dim=opt.feat_dim, in_channels=in_channels)
    else:
        model = SupConResNet_inter(name=opt.model, feat_dim=opt.feat_dim, in_channels=in_channels)

    model = load_model(opt, model)

    return model


def load_model(opt, model=None):

    ckpt = torch.load(opt.model_path, map_location='cpu')
    state_dict = ckpt['model']

    new_state_dict = {}
    for k, v in state_dict.items():
        k = k.replace("module.", "")
        new_state_dict[k] = v

    state_dict = new_state_dict
    model.load_state_dict(state_dict)
    model.eval()

    return model


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

        # print(idx)
        # images = images.cuda(non_blocking=True)
        # labels = labels.cuda(non_blocking=True)
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

    # testing_features = testing_features.astype(np.double)
    # testing_features = testing_features / np.linalg.norm(testing_features, axis=1)[:, np.newaxis]  ####

    for idx, testing_feature in enumerate(testing_features):
        # print(idx)
        similarity_logits = []
        for training_features_c in sorted_exemplar_features:
            training_features_c = np.array(training_features_c, dtype=float)
            # training_features_c = training_features_c[::2]                                                          # TODO

            similarities = np.matmul(training_features_c, testing_feature) / np.linalg.norm(training_features_c,
                                                                                            axis=1) / np.linalg.norm(
                testing_feature)
            ind = np.argsort(similarities)[-opt.K:]
            top_k_similarities = similarities[ind]
            similarity_logits.append(np.sum(top_k_similarities))  # !!!!
            # similarity_logits.append(top_k_similarities[-1])

        testing_similarity_logits.append(similarity_logits)

    testing_similarity_logits = np.array(testing_similarity_logits)
    testing_similarity_logits = np.divide(testing_similarity_logits.T, np.sum(testing_similarity_logits,
                                                                              axis=1)).T  # normalization, maybe not necessary???

    return testing_similarity_logits



def distances(stats, test_features, mode="mahalanobis"):
    dis_logits_out = []
    dis_logits_in = []
    dis_preds = []
    for features in test_features:
        diss = []
        for i, (mu, var) in enumerate(stats):
            # mu, var = stats[0]                             ##### delete
            if mode == "mahalanobis":
                features_normalized = features - mu
                # dis =  np.matmul(features_normalized, np.linalg.inv(var))
                # dis = np.matmul(dis, np.swapaxes(features_normalized, 0, 1))
                # dis = dis[0][0]
                if np.linalg.matrix_rank(var) < var.shape[0]:
                    dis = mahalanobis(features, mu, np.linalg.pinv(var))
                else:
                    dis = mahalanobis(features, mu, np.linalg.inv(var))
            else:
                features = np.squeeze(np.array(features))
                dis = features - mu
                dis = np.sum(np.abs(dis))

            diss.append(dis)

        dis_logits_out.append(np.min(np.array(diss)) / np.sum(np.array(
            diss)))  # !!!!!!!!!!!!!!!!!! minus here !!!!!!!!!!!! to entsprechen 0 for outliers and 1 for inliers, unknown logits, flip for known logits
        dis_logits_in.append(-np.min(np.array(diss)))
        dis_preds.append(np.argmin(np.array(diss)))

    return dis_logits_in, dis_logits_out, dis_preds


def KNN_classifier(testing_features, testing_labels, sorted_training_features):
    print("Begin KNN Classifier!")
    testing_similarity_logits = KNN_logits(testing_features, sorted_training_features)
    prediction_logits, predictions = np.amax(testing_similarity_logits, axis=1), np.argmax(testing_similarity_logits,
                                                                                           axis=1)
    # prediction_logits, predictions = -np.amin(testing_similarity_logits, axis=1), np.argmin(testing_similarity_logits, axis=1)       # minus here, larger score for inliers

    acc = accuracy_plain(predictions, testing_labels)
    print("KNN Accuracy is: ", acc)

    return prediction_logits, predictions, acc


def distance_classifier(testing_features, testing_labels, sorted_training_features):
    stats = feature_stats(sorted_training_features)
    dis_logits_in, dis_logits_out, dis_preds = distances(stats, testing_features)

    acc = accuracy_plain(dis_preds, testing_labels)
    print("Distance Accuracy is: ", acc)

    return dis_logits_in, dis_logits_out, dis_preds, acc


def feature_classifier(opt):
    print(opt.exemplar_features_path)
    with open(opt.exemplar_features_path, "rb") as f:
        features_exemplar_head1, features_exemplar_head2, features_exemplar_head3, features_exemplar_backbone, labels_examplar = pickle.load(f)
        labels_examplar = np.squeeze(np.array(labels_examplar))

    if opt.testing_known_features_path is not None:
        with open(opt.testing_known_features_path, "rb") as f:
            features_testing_known_head1, features_testing_known_head2, features_testing_known_head3, features_testing_known_backbone, labels_testing_known = pickle.load(f)
            labels_testing_known = np.squeeze(np.array(labels_testing_known))

    with open(opt.testing_unknown_features_path, "rb") as f:
        features_testing_unknown_head1, features_testing_unknown_head2, features_testing_unknown_head3, features_testing_unknown_backbone, labels_testing_unknown = pickle.load(
            f)
        labels_testing_unknown = np.squeeze(np.array(labels_testing_unknown))

    labels_binary_known = [1 if i < 100 else 0 for i in labels_testing_known]
    labels_binary_unknown = [1 if i < 100 else 0 for i in labels_testing_unknown]
    labels_binary = np.array(labels_binary_known + labels_binary_unknown)

    model = set_model(opt)
    features_testing_known_backbone = torch.tensor(features_testing_known_backbone.astype(np.float32))
    features_testing_known_head1 = model.head1(features_testing_known_backbone)
    features_testing_known_head2 = model.head2(features_testing_known_backbone)
    features_testing_known_head3 = model.head3(features_testing_known_backbone)
    features_testing_known_head_cat = torch.cat((features_testing_known_head1, features_testing_known_head2, features_testing_known_head3), dim=1)
    features_testing_known_head_sum = features_testing_known_head1 + features_testing_known_head2 + features_testing_known_head3

    features_testing_unknown_backbone = torch.tensor(features_testing_unknown_backbone.astype(np.float32))
    features_testing_unknown_head1 = model.head1(features_testing_unknown_backbone)
    features_testing_unknown_head2 = model.head2(features_testing_unknown_backbone)
    features_testing_unknown_head3 = model.head3(features_testing_unknown_backbone)
    features_testing_unknown_head_cat = torch.cat((features_testing_unknown_head1, features_testing_unknown_head2, features_testing_unknown_head3), dim=1)
    features_testing_unknown_head_sum = features_testing_unknown_head1 + features_testing_unknown_head2 + features_testing_unknown_head3

    norm_score_known1 = np.linalg.norm(features_testing_known_head1.detach().numpy(), axis=1)
    norm_score_unknown1 = np.linalg.norm(features_testing_unknown_head1.detach().numpy(), axis=1)
    norm_score_binary1 = np.concatenate((norm_score_known1, norm_score_unknown1), axis=0)
    auroc = AUROC(labels_binary, norm_score_binary1, opt)
    print("AUROC norm 1 is: ", auroc)
    
    norm_score_known2 = np.linalg.norm(features_testing_known_head2.detach().numpy(), axis=1)
    norm_score_unknown2 = np.linalg.norm(features_testing_unknown_head2.detach().numpy(), axis=1)
    norm_score_binary2 = np.concatenate((norm_score_known2, norm_score_unknown2), axis=0)
    auroc = AUROC(labels_binary, norm_score_binary2, opt)
    print("AUROC norm 2 is: ", auroc)
    
    norm_score_known3 = np.linalg.norm(features_testing_known_head3.detach().numpy(), axis=1)
    norm_score_unknown3 = np.linalg.norm(features_testing_unknown_head3.detach().numpy(), axis=1)
    norm_score_binary3 = np.concatenate((norm_score_known3, norm_score_unknown3), axis=0)
    auroc = AUROC(labels_binary, norm_score_binary3, opt)
    print("AUROC norm 3 is: ", auroc)
    
    norm_score_known_cat = np.linalg.norm(features_testing_known_head_cat.detach().numpy(), axis=1)
    norm_score_unknown_cat = np.linalg.norm(features_testing_unknown_head_cat.detach().numpy(), axis=1)
    norm_score_binary_cat = np.concatenate((norm_score_known_cat, norm_score_unknown_cat), axis=0)
    auroc = AUROC(labels_binary, norm_score_binary_cat, opt)
    print("AUROC norm cat is: ", auroc)
    
    norm_score_known_sum = np.linalg.norm(features_testing_known_head_sum.detach().numpy(), axis=1)
    norm_score_unknown_sum = np.linalg.norm(features_testing_unknown_head_sum.detach().numpy(), axis=1)
    norm_score_binary_sum = np.concatenate((norm_score_known_sum, norm_score_unknown_sum), axis=0)
    auroc = AUROC(labels_binary, norm_score_binary_sum, opt)
    print("AUROC norm sum is: ", auroc)
    
    norm_score_known_score_sum = norm_score_known1 + norm_score_known2 + norm_score_known3
    norm_score_unknown_score_sum = norm_score_unknown1 + norm_score_unknown2 + norm_score_unknown3
    norm_score_binary_score_sum = np.concatenate((norm_score_known_score_sum, norm_score_unknown_score_sum), axis=0)
    auroc = AUROC(labels_binary, norm_score_binary_score_sum, opt)
    print("AUROC norm score sum is: ", auroc)
    
    res12_known = stats.pearsonr(norm_score_known1, norm_score_known2)
    res13_known = stats.pearsonr(norm_score_known1, norm_score_known3)
    res23_known = stats.pearsonr(norm_score_known2, norm_score_known3)
    print("res12_known", res12_known)
    print("res13_known", res13_known)
    print("res23_known", res23_known)
    
    res12_unknown = stats.pearsonr(norm_score_unknown1, norm_score_unknown2)
    res13_unknown = stats.pearsonr(norm_score_unknown1, norm_score_unknown3)
    res23_unknown = stats.pearsonr(norm_score_unknown2, norm_score_unknown3)
    print("res12_unknown", res12_unknown)
    print("res13_unknown", res13_unknown)
    print("res23_unknown", res23_unknown)
    

    """
    features_testing_known_head, labels_testing_known = down_sampling(features_testing_known_head, labels_testing_known,
                                                                      opt.downsampling_ratio_known)
    prediction_logits_known, predictions_known, acc_known = KNN_classifier(features_testing_known_head,
                                                                           labels_testing_known,
                                                                           sorted_features_examplar_head)

    prediction_logits_known_dis_in, prediction_logits_known_dis_out, predictions_known_dis, acc_known_dis = distance_classifier(
        features_testing_known_head, labels_testing_known, sorted_features_examplar_head)

   

    features_testing_unknown_head, labels_testing_unknown = down_sampling(features_testing_unknown_head,
                                                                          labels_testing_unknown,
                                                                          opt.downsampling_ratio_unknown)

    prediction_logits_unknown, predictions_unknown, _ = KNN_classifier(features_testing_unknown_head,
                                                                       labels_testing_unknown,
                                                                       sorted_features_examplar_head)

    prediction_logits_unknown_dis_in, prediction_logits_unknown_dis_out, predictions_unknown_dis, acc_unknown_dis = distance_classifier(
        features_testing_unknown_head, labels_testing_unknown, sorted_features_examplar_head)

    knn_predictions = np.concatenate((predictions_known, predictions_unknown), axis=0)
    distance_predictions = np.concatenate((predictions_known_dis, predictions_unknown_dis), axis=0)
    labels_testing = np.concatenate((labels_testing_known, labels_testing_unknown), axis=0)

    with open(opt.prediction_save_path, "wb") as f:
        pickle.dump((knn_predictions, distance_predictions, labels_testing), f)

    # Process results AUROC and OSCR
    # for AUROC, convert labels to binary labels, assume inliers are positive
    labels_binary_known = [1 if i < 100 else 0 for i in labels_testing_known]
    labels_binary_unknown = [1 if i < 100 else 0 for i in labels_testing_unknown]
    labels_binary = np.array(labels_binary_known + labels_binary_unknown)

    probs_binary = np.concatenate((prediction_logits_known, prediction_logits_unknown), axis=0)

    auroc = AUROC(labels_binary, probs_binary, opt)
    print("AUROC is: ", auroc)

    probs_binary_dis = np.concatenate((prediction_logits_known_dis_in, prediction_logits_unknown_dis_in), axis=0)
    # print("probs_binary", probs_binary_dis)

    auroc = AUROC(labels_binary, probs_binary_dis, opt)
    print("Dis AUROC is: ", auroc)

    # OSCR
    oscr = OSCR(np.array(prediction_logits_known_dis_out), np.array(prediction_logits_unknown_dis_out),
                predictions_known, labels_testing_known)
    print("OSCR is: ", oscr)
    """

    return auroc


if __name__ == "__main__":
    opt = parse_option()

    auroc = feature_classifier(opt)

