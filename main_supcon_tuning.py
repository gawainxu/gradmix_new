from __future__ import print_function

import os
import sys
import argparse
import time
import math
import pickle
import numpy as np

import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn

from util import TwoCropTransform, AverageMeter
from util import adjust_learning_rate, save_model
from datautil import vanilla_mixup, salient_cutmix, vanilla_cutmix
from datautil import num_inlier_classes_mapping
from networks.resnet_big import SupConResNet, LinearClassifier
from networks.resnet_big import ResNet_Pretrain
from networks.resnet_big import MoCoResNet
from networks.maskcon import MaskCon
from networks.simCNN import simCNN_contrastive
from networks.resnet_preact import SupConpPreactResNet
from networks.mlp import SupConMLP
from networks.vgg import SupConVGG
from losses import SupConLoss
from datautil import get_train_datasets, get_test_datasets
from  data_loader import safe_collate

import matplotlib

matplotlib.use('Agg')

try:
    import apex
    from apex import amp, optimizers
except ImportError:
    pass


def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--print_freq', type=int, default=10,
                        help='print frequency')
    parser.add_argument('--save_freq', type=int, default=50,
                        help='save frequency')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='batch_size')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of training epochs')
    parser.add_argument("--grad_splits", type=int, default=1)

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='1000',
                        help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1,
                        help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum')
    parser.add_argument("--pretrained", type=int, default=1)

    # model dataset
    parser.add_argument('--model', type=str, default='resnet50_pretrain',
                        choices=["resnet18", "resnet34", "resnet50_pretrain", "resnet18_pretrain"])
    parser.add_argument('--datasets', type=str, default='cifar10', help='dataset')
    parser.add_argument('--mean', type=str, help='mean of dataset in path in form of str tuple')
    parser.add_argument('--std', type=str, help='std of dataset in path in form of str tuple')
    parser.add_argument('--data_folder', type=str, default=None, help='path to custom dataset')
    parser.add_argument('--size', type=int, default=32, help='parameter for RandomResizedCrop')
    parser.add_argument("--augmentation_list", type=list, default=[])
    parser.add_argument("--argmentation_n", type=int, default=1)
    parser.add_argument("--argmentation_m", type=int, default=6)
    parser.add_argument("--augmix", type=bool, default=False)

    # method
    parser.add_argument('--method', type=str, default='SimCLR',
                        choices=['SupCon', 'SimCLR', "SimCLR_CE", "MoCo"], help='choose method')
    parser.add_argument("--method_gama", type=float, default=1.0)
    parser.add_argument("--method_lam", type=float, default=1.0)
    parser.add_argument("--trail", type=int, default=0, choices=[0, 1, 2, 3, 4, 5, 6],
                        help="index of repeating training")
    parser.add_argument("--action", type=str, default="training_supcon",
                        choices=["training_supcon", "trainging_linear", "testing_known", "testing_unknown",
                                 "feature_reading"])
    # temperature
    parser.add_argument('--temp', type=float, default=0.05, help='temperature for loss')

    # other setting
    parser.add_argument('--cosine', type=bool, default=False,
                        help='using cosine annealing')
    parser.add_argument('--syncBN', action='store_true',
                        help='using synchronized batch normalization')
    parser.add_argument('--warm', action='store_true',
                        help='warm-up for large batch training')
    parser.add_argument("--augmentation_method", type=str, default="vanilia",
                        choices=["vanilia", "mixup_negative", "mixup_positive", "mixup_hybrid", "mixup_vanilla",
                                 "mixup_vanilla_features"])
    parser.add_argument("--data_method", type=str, default="original",
                        choices=["original", "upsampling"])
    parser.add_argument("--architecture", type=str, default="single", choices=["single", "multi"])
    parser.add_argument("--ensemble_num", type=int, default=1)
    parser.add_argument("--feat_dim", type=int, default=128)
    parser.add_argument("--grad_layers", type=str, default="0")
    parser.add_argument("--old_augmented", type=bool, default=True)
    parser.add_argument("--supcon_aug", type=int, default=0)
    parser.add_argument("--frozen_layers", type=int, default=-1)

    # moco parameters
    parser.add_argument("--K", type=int, default=4096, help="buffer size in moco")
    parser.add_argument("--momentum_moco", type=float, default=0.999)

    # upsampling parameters for feature boundary based augmentation
    parser.add_argument("--upsample", type=bool, default=False)
    parser.add_argument("--portion_out", type=float, default=0.5)
    parser.add_argument("--upsample_times", type=int, default=1)
    parser.add_argument("--last_feature_path", type=str, default=None)
    parser.add_argument("--last_model_path", type=str, default=None)

    # mixup parameters
    parser.add_argument("--alpha_negative", type=float, default=0.2, help="between 0.2 to 0.4")
    parser.add_argument("--alpha_positive", type=float, default=0.2, help="between 0.2 to 0.4")
    parser.add_argument("--alpha_hybrid", type=float, default=10)
    parser.add_argument("--beta_hybrid", type=float, default=0.3)
    parser.add_argument("--alpha_vanilla", type=float, default=10)
    parser.add_argument("--beta_vanilla", type=float, default=0.3)
    parser.add_argument("--intra_inter_mix_positive", type=bool, default=False, help="intra=True, inter=False")
    parser.add_argument("--intra_inter_mix_negative", type=bool, default=False, help="intra=True, inter=False")
    parser.add_argument("--intra_inter_mix_hybrid", type=bool, default=False, help="intra=True, inter=False")
    parser.add_argument("--mixup_positive", type=bool, default=False)
    parser.add_argument("--mixup_negative", type=bool, default=False)
    parser.add_argument("--mixup_hybrid", type=bool, default=False)
    parser.add_argument("--mixup_vanilla", type=bool, default=False)
    parser.add_argument("--mixup_vanilla_features", type=bool, default=False)
    parser.add_argument("--positive_p", type=float, default=0.5)
    parser.add_argument("--alfa", type=float, default=1)
    parser.add_argument("--positive_method", type=str, default="layersaliencymix",
                        choices=["max_similarity", "min_similarity", "random", "prob_similarity", "reverse",
                                 "saliencymix", "layersaliencymix", "cutmix", "cv2saliency"])
    parser.add_argument("--negative_method", type=str, default="no", choices=["max_similarity", "random", "even", "no"])
    parser.add_argument("--hybrid_method", type=str, default="no", choices=["min_similarity", "random", "no"])
    parser.add_argument("--vanilla_method", type=str, default="no", choices=["reverse", "random", "no"])
    parser.add_argument("--randaug", type=int, default=0)
    parser.add_argument("--apool", type=bool, default=False)
    parser.add_argument("--use_cuda", type=bool, default=True)
    parser.add_argument("--record_grad", type=bool, default=False)

    opt = parser.parse_args()

    # check if dataset is path that passed required arguments
    if opt.datasets == 'path':
        assert opt.data_folder is not None \
               and opt.mean is not None \
               and opt.std is not None

    # set the path according to the environment
    if opt.data_folder is None:
        opt.data_folder = '../datasets/'
    opt.model_path = './save/SupCon/{}_models'.format(opt.datasets)
    print("opt.model_path", opt.model_path)
    if not os.path.isdir(opt.model_path):
        os.mkdir(opt.model_path)

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    opt.model_name = opt.datasets + "_" + opt.model

    if opt.augmentation_method == "mixup_positive":
        opt.model_name += "_mixup_positive_" + "alpha_" + str(opt.alpha_vanilla) + "_beta_" + str(
            opt.beta_vanilla) + "_" + opt.positive_method + "_" + opt.grad_layers
    elif opt.augmentation_method == "vanilia":
        opt.model_name += "_vanilia_"

    if opt.method == "SupCon":
        opt.model_name += "_SupCon_"
    if opt.method == "MoCo":
        opt.model_name += "_MoCo_"
    elif opt.method == "SimCLR":
        opt.model_name += "_SimCLR_" + str(opt.method_gama) + "_" + str(opt.method_lam) + "_" + str(opt.temp) + "_"
    elif opt.method == "SimCLR_CE":
        opt.model_name += "_SimCLR_CE_" + str(opt.method_lam) + "_"

    opt.model_name += 'trail_{}'.format(opt.trail) + "_" + str(opt.feat_dim) + "_" + str(opt.batch_size)

    if opt.last_model_path is not None:
        opt.model_name += "_twostage2"

    if opt.augmix:
        opt.model_name += "_augmix"

    if opt.old_augmented:
        opt.model_name += "_old_augmented"

    if opt.supcon_aug:
        opt.model_name += "_supcon_aug"

    opt.model_name += "_finetune"

    # warm-up for large-batch training,
    if opt.batch_size > 256:
        opt.warm = True
    if opt.warm:
        opt.model_name = '{}_warm'.format(opt.model_name)
        opt.warmup_from = 0.01
        opt.warm_epochs = 10
        if opt.cosine:
            eta_min = opt.learning_rate * (opt.lr_decay_rate ** 3)
            opt.warmup_to = eta_min + (opt.learning_rate - eta_min) * (
                    1 + math.cos(math.pi * opt.warm_epochs / opt.epochs)) / 2
        else:
            opt.warmup_to = opt.learning_rate

    opt.save_folder = os.path.join(opt.model_path, opt.model_name)
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)

    opt.num_classes = num_inlier_classes_mapping[opt.datasets]

    layers = opt.grad_layers.split(",")
    opt.grad_layers = list([])
    for l in layers:
        opt.grad_layers.append(int(l))

    return opt


def set_loader(opt):
    # construct data loader

    if opt.upsample is True and opt.last_feature_path is not None:

        last_features_list = []
        last_feature_labels_list = []

        with open(opt.last_feature_path, "rb") as f:
            last_features, _, _, last_feature_labels = pickle.load(f)
            last_features_list.append(last_features)
            last_feature_labels_list.append(last_feature_labels)

        last_model = load_model(opt)
        train_dataset = get_train_datasets(opt, last_features_list=last_features_list,
                                           last_feature_labels_list=last_feature_labels_list, last_model=last_model)
    else:
        train_dataset = get_train_datasets(opt)
        test_dataset = get_test_datasets(opt)

    train_sampler = None
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True,
                                               num_workers=opt.num_workers, pin_memory=True, sampler=train_sampler,
                                               drop_last=True,
                                               persistent_workers=True, collate_fn=safe_collate)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1,
                                              shuffle=False, num_workers=opt.num_workers, pin_memory=True,
                                              sampler=train_sampler, drop_last=True, persistent_workers=True, collate_fn=safe_collate)

    return train_loader, test_loader


def set_model(opt):
    if opt.datasets == "mnist":
        in_channels = 1
    elif opt.datasets == "FUB":
        in_channels = 1
    else:
        in_channels = 3

    if opt.method == "MoCo":
        model = MoCoResNet(opt, name=opt.model, feat_dim=opt.feat_dim, in_channels=in_channels)
        criterion1 = torch.nn.CrossEntropyLoss()
        criterion2 = SupConLoss(temperature=opt.temp)
        linear = None
    elif opt.method == "SimCLR_CE":
        model = SupConResNet(name=opt.model, feat_dim=opt.feat_dim, in_channels=in_channels)
        criterion1 = SupConLoss(temperature=opt.temp)
        criterion2 = torch.nn.CrossEntropyLoss()
        linear = LinearClassifier(name=opt.model, num_classes=opt.num_classes)
    elif opt.method == "MaskCon":
        model = MaskCon(arch="resnet18", T1=opt.method_T1, T2=opt.method_T2)
        criterion1 = None
        criterion2 = None
        linear = None
    elif opt.method == "SimCLR" or opt.method == "SupCon":
        if opt.model == "resnet18_pretrain":
            model = ResNet_Pretrain(name='resnet18', feat_dim=opt.feat_dim, freeze_layers=opt.frozen_layers)
        elif opt.model == "resnet50_pretrain":
            model = ResNet_Pretrain(name='resnet50', feat_dim=opt.feat_dim, freeze_layers=opt.frozen_layers)
        else:
            model = simCNN_contrastive(opt, feature_dim=opt.feat_dim, in_channels=in_channels)

        criterion1 = SupConLoss(temperature=opt.temp)
        criterion2 = SupConLoss(temperature=opt.temp)
        linear = None

    if opt.last_model_path is not None:
        print("model loaded")
        load_model(opt, model)

    # enable synchronized Batch Normalization
    if opt.syncBN:
        model = apex.parallel.convert_syncbn_model(model)

    if torch.cuda.is_available() and opt.use_cuda is True:

        if torch.cuda.device_count() > 1:
            if opt.method == "MoCo" or opt.method == "MaskCon":
                model.encoder_k = torch.nn.DataParallel(model.encoder_k)
                model.encoder_q = torch.nn.DataParallel(model.encoder_q)
            else:
                model.encoder = torch.nn.DataParallel(model.encoder)

        model = model.cuda()
        if linear is not None:
            linear = linear.cuda()
        if criterion1 is not None:
            criterion1 = criterion1.cuda()
        if criterion2 is not None:
            criterion2 = criterion2.cuda()
        cudnn.benchmark = True

    return model, linear, criterion1, criterion2


def load_model(opt, model=None):
    if model is None:
        model = SupConResNet(name=opt.model)

    ckpt = torch.load(opt.last_model_path, map_location='cpu')
    state_dict = ckpt['model']

    new_state_dict = {}
    for k, v in state_dict.items():
        k = k.replace("module.", "")
        new_state_dict[k] = v

    state_dict = new_state_dict
    model.load_state_dict(state_dict)
    if torch.cuda.is_available() and opt.use_cuda is True:
        model.cuda()
    model.eval()

    return model


def set_optimizer(opt, model):

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                           lr=opt.learning_rate)
    return optimizer


def train(train_loader, model, linear, criterion1, criterion2, optimizer, epoch, opt):
    """one epoch training"""
    #model.train()
    model.encoder.eval()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    losses_ssl = AverageMeter()
    losses_sup = AverageMeter()
    losses1 = AverageMeter()
    losses2 = AverageMeter()
    losses_ssl_mix = AverageMeter()
    losses_supcon_mix = AverageMeter()

    end = time.time()
    ious_epoch = []
    loss_sl_grad = []
    loss_ssl_grad = []
    loss_sl_hessian = []
    loss_ssl_hessian = []

    for idx, (images, labels) in enumerate(train_loader):

        if images is None:
            continue

        data_time.update(time.time() - end)
        images1 = images[0]
        images2 = images[1]

        if opt.method == "SimCLR_CE":
            image3 = images[2]

        if torch.cuda.is_available() and opt.use_cuda is True:

            images1 = images1.cuda(non_blocking=True)
            images2 = images2.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)

        images = torch.cat([images1, images2], dim=0)
        bsz = labels.shape[0]

        # warm-up learning rate
        # warmup_learning_rate(opt, epoch, idx, len(train_loader), optimizer)

        if linear is not None:
            labels_linear = labels  # torch.cat([labels, labels], dim=0)
            if torch.cuda.is_available() and opt.use_cuda is True:
                image3 = image3.cuda(non_blocking=True)
            # labels_linear = labels_linear.cuda(non_blocking=True)
            logits = linear(model.encoder(image3))

        if opt.method == 'SupCon':
            features = model(images)
            features1, features2 = torch.split(features, [bsz, bsz], dim=0)
            features = torch.cat([features1.unsqueeze(1), features2.unsqueeze(1)], dim=1)
            loss1 = criterion1(features, labels)
            loss = loss2 = loss1
            loss_sup = loss
            loss_ssl = torch.tensor([0])

            if opt.mixup_positive:
                if opt.positive_method == "cutmix" or opt.positive_method == "snapmix":
                    mixed_positive_samples1, mixed_positive_samples2, lam = vanilla_cutmix(images1, images2, opt)
                elif opt.positive_method == "saliencymix":
                    mixed_positive_samples1, mixed_positive_samples2, lam = salient_cutmix(images1, images2, model, opt)
                elif opt.positive_method == "layersaliencymix":
                    mixed_positive_samples1, mixed_positive_samples2, lam = salient_cutmix(images1, images2, model, opt)
                elif opt.positive_method == "attentive_mix":
                    mixed_positive_samples1, mixed_positive_samples2, lam = attentive_cutmix(images1, images2, opt)
                elif opt.positive_method == "cv2saliency":
                    mixed_positive_samples1, mixed_positive_samples2, lam = salient_cutmix(images1, images2, model, opt)
                else:
                    mixed_positive_samples1, mixed_positive_samples2, labels_new, lam = vanilla_mixup(images1, images2,
                                                                                                      labels,
                                                                                                      alpha=opt.alpha_vanilla,
                                                                                                      beta=opt.beta_vanilla,
                                                                                                      mode=opt.positive_method,
                                                                                                      encoder=model)

                mixed_positive_samples = torch.cat([mixed_positive_samples1, mixed_positive_samples2], dim=0)
                mixed_positive_features = model(mixed_positive_samples)
                mixed_positive_features1, mixed_positive_features2 = torch.split(mixed_positive_features,
                                                                                 [bsz, bsz], dim=0)
                mixed_positive_features = torch.cat(
                    [mixed_positive_features1.unsqueeze(1), mixed_positive_features2.unsqueeze(1)], dim=1)
                loss_supcon_mix = criterion1(features, labels=labels,
                                             features_positive=mixed_positive_features)  # criterion1(mixed_positive_features) #  !!!!!!!!!! TODO

                if opt.old_augmented:
                    loss_supcon = loss_supcon + lam * loss_supcon_mix
                    losses_supcon_mix.update(loss_supcon_mix.detach().cpu().item())

        elif opt.method == 'SimCLR':
            features = model(images)
            features1, features2 = torch.split(features, [bsz, bsz], dim=0)
            features = torch.cat([features1.unsqueeze(1), features2.unsqueeze(1)], dim=1)

            loss_sup = loss1 = criterion2(features, labels)
            loss_ssl = loss2 = criterion1(features)

            if opt.mixup_positive:
                if opt.positive_method == "cutmix" or opt.positive_method == "snapmix":
                    mixed_positive_samples1, mixed_positive_samples2, lam = vanilla_cutmix(images1, images2, opt)
                elif opt.positive_method == "saliencymix":
                    mixed_positive_samples1, mixed_positive_samples2, lam = salient_cutmix(images1, images2, model, opt)
                elif opt.positive_method == "layersaliencymix":
                    mixed_positive_samples1, mixed_positive_samples2, lam = salient_cutmix(images1, images2, model, opt)
                elif opt.positive_method == "attentive_mix":
                    mixed_positive_samples1, mixed_positive_samples2, lam = attentive_cutmix(images1, images2, opt)
                elif opt.positive_method == "cv2saliency":
                    mixed_positive_samples1, mixed_positive_samples2, lam = salient_cutmix(images1, images2, model, opt)
                else:
                    mixed_positive_samples1, mixed_positive_samples2, labels_new, lam = vanilla_mixup(images1,
                                                                                                      images2,
                                                                                                      labels,
                                                                                                      alpha=opt.alpha_vanilla,
                                                                                                      beta=opt.beta_vanilla,
                                                                                                      mode=opt.positive_method,
                                                                                                      encoder=model)

                mixed_positive_samples = torch.cat([mixed_positive_samples1, mixed_positive_samples2], dim=0)
                mixed_positive_features = model(mixed_positive_samples)
                mixed_positive_features1, mixed_positive_features2 = torch.split(mixed_positive_features,
                                                                                 [bsz, bsz], dim=0)
                mixed_positive_features = torch.cat(
                    [mixed_positive_features1.unsqueeze(1), mixed_positive_features2.unsqueeze(1)], dim=1)
                loss_ssl_mix = criterion1(features,
                                          features_positive=mixed_positive_features)  # criterion1(mixed_positive_features) #  !!!!!!!!!! TODO

                if opt.supcon_aug:
                    loss_supcon_mix = criterion2(features, labels, features_positive=mixed_positive_features)
                    loss_sup = loss1 = loss_sup + lam * loss_supcon_mix
                    losses_supcon_mix.update(loss_supcon_mix.detach().cpu().item())

                if opt.old_augmented:
                    loss_ssl = loss_ssl + lam * loss_ssl_mix
                else:
                    loss_ssl = loss2 = loss_ssl_mix
                losses_ssl_mix.update(loss_ssl_mix.detach().cpu().item())

            loss = opt.method_gama * loss_sup + opt.method_lam * loss_ssl
            losses_ssl.update(loss_ssl.detach().cpu().item())
            losses_sup.update(loss_sup.detach().cpu().item())

            if opt.record_grad:
                # Here the model parameters can be other intermdiate parameters
                g_ssl = torch.autograd.grad(loss_ssl, [model.head[0].weight, model.head[2].weight],
                                            retain_graph=True)  # model.encoder.layer4[-1].conv2.weight,
                g_sl = torch.autograd.grad(loss_sup, [model.head[0].weight, model.head[2].weight],
                                           retain_graph=True)  # model.encoder.layer4[-1].conv2.weight,
                loss_ssl_grad.append([x.detach().cpu() for x in g_ssl])
                loss_sl_grad.append([x.detach().cpu() for x in g_sl])
                print("g_sup", torch.sum(torch.abs(loss_sl_grad[-1][0])),
                      torch.sum(torch.abs(loss_sl_grad[-1][1])))
                print("g_ssl", torch.sum(torch.abs(loss_ssl_grad[-1][0])),
                      torch.sum(torch.abs(loss_ssl_grad[-1][1])))
                # loss_ssl_hessian.append([x.detach().cpu() for x in hessian_ssl])
                # loss_sl_hessian.append([x.detach().cpu() for x in hessian_sl])

        elif opt.method == 'SimCLR_CE':
            features = model(images)
            features1, features2 = torch.split(features, [bsz, bsz], dim=0)
            features = torch.cat([features1.unsqueeze(1), features2.unsqueeze(1)], dim=1)
            loss_ce = loss1 = criterion2(logits, labels_linear)
            loss_ssl = loss2 = criterion1(features)
            loss = opt.method_gama * loss_ce + opt.method_lam * loss_ssl

            losses_ssl.update(loss_ssl.detach().cpu().item())
            losses_sup.update(loss_ce.detach().cpu().item())

        elif opt.method == "MoCo":
            logits, labels_moco = model(images1, images2, mode="moco")
            loss_moco = loss2 = criterion1(logits, labels_moco)

            if opt.mixup_positive:
                if opt.positive_method == "cutmix":
                    mixed_positive_samples1, mixed_positive_samples2, lam = vanilla_cutmix(images1, images2, opt)
                elif opt.positive_method == "saliencymix":
                    mixed_positive_samples1, mixed_positive_samples2, lam = salient_cutmix(images1, images2, model, opt)
                elif opt.positive_method == "layersaliencymix":
                    mixed_positive_samples1, mixed_positive_samples2, lam = salient_cutmix(images1, images2, model, opt)
                else:
                    mixed_positive_samples1, mixed_positive_samples2, labels_new, lam = vanilla_mixup(images1, images2,
                                                                                                      labels,
                                                                                                      alpha=opt.alpha_vanilla,
                                                                                                      beta=opt.beta_vanilla,
                                                                                                      mode=opt.positive_method,
                                                                                                      encoder=model)
                logits_mix, labels_moco_mix = model(mixed_positive_samples1, mixed_positive_samples2, mode="moco")
                loss_moco_mix = criterion1(logits_mix, labels_moco_mix)
                loss_moco = loss2 = loss_moco + lam * loss_moco_mix

            loss1 = loss2
            loss = opt.method_lam * loss_moco  # opt.method_gama * loss_sup +

        else:
            raise ValueError('contrastive method not supported: {}'.
                             format(opt.method))

        # update metric
        losses.update(loss.detach().cpu().item(), bsz)
        losses1.update(loss_sup.detach().cpu().item(), bsz)
        losses2.update(loss_ssl.detach().cpu().item(), bsz)

        # SGD
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if (idx + 1) % opt.print_freq == 0:
            print('Train: [{0}][{1}/{2}]\t'
                  'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'loss {loss.val:.3f} ({loss.avg:.3f})\t'
                  "loss1 {loss1.val:.3f} ({loss1.avg:.3f})\t"
                  "loss2 {loss2.val:.3f} ({loss2.avg:.3f})\t".format(
                epoch, idx + 1, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, loss1=losses1, loss2=losses2))
            if opt.method == "SimCLR":
                print("loss_ssl {loss_ssl.val:.3f} ({loss_ssl.avg:.3f})\t"
                      "loss_sup {loss_sup.val:.3f} ({loss_sup.avg:.3f})\t".format(
                    loss_ssl=losses_ssl, loss_sup=losses_sup))
                if opt.mixup_positive:
                    print("loss_ssl_mix {loss_ssl_mix.val:.3f} ({loss_ssl_mix.avg:.3f})\t".format(
                        loss_ssl_mix=losses_ssl_mix))
            if opt.method == "SimCLR_CE":
                print("loss_ssl {loss_ssl.val:.3f} ({loss_ssl.avg:.3f})\t"
                      "loss_sup {loss_ce.val:.3f} ({loss_ce.avg:.3f})\t".format(
                    loss_ssl=losses_ssl, loss_ce=losses_sup))
            sys.stdout.flush()

    return (losses.avg, losses1.avg, losses2.avg), ious_epoch, (
    loss_ssl_grad, loss_sl_grad)  # , (loss_ssl_hessian, loss_sl_hessian)


def validate(vali_loader, model, linear, criterion1, criterion2, optimizer, epoch, opt):
    """one epoch training"""

    model.eval()
    losses = AverageMeter()
    losses_ssl = AverageMeter()
    losses_sup = AverageMeter()
    losses1 = AverageMeter()
    losses2 = AverageMeter()

    end = time.time()
    ious_epoch = []

    for idx, (images, labels) in enumerate(vali_loader):

        if images is None:
            continue

        images1 = images[0]
        images2 = images[1]

        if opt.method == "SimCLR_CE":
            image3 = images[2]

        images = torch.cat([images1, images2], dim=0)

        if torch.cuda.is_available() and opt.use_cuda is True:
            images = images.cuda(non_blocking=True)
            images1 = images1.cuda(non_blocking=True)
            images2 = images2.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)

        bsz = labels.shape[0]

        if linear is not None:
            labels_linear = labels  # torch.cat([labels, labels], dim=0)
            if torch.cuda.is_available() and opt.use_cuda is True:
                image3 = image3.cuda(non_blocking=True)
            # labels_linear = labels_linear.cuda(non_blocking=True)
            logits = linear(model.encoder(image3))

        if opt.method == 'SupCon':

            features = model(images)
            features1, features2 = torch.split(features, [bsz, bsz], dim=0)
            features = torch.cat([features1.unsqueeze(1), features2.unsqueeze(1)], dim=1)
            loss1 = criterion1(features, labels)
            loss = loss2 = loss1

        elif opt.method == 'SimCLR':
            features = model(images)
            features1, features2 = torch.split(features, [bsz, bsz], dim=0)
            features = torch.cat([features1.unsqueeze(1), features2.unsqueeze(1)], dim=1)

            loss_sup = loss1 = criterion2(features, labels)
            loss_ssl = loss2 = criterion1(features)

            if opt.mixup_positive:
                losses_ssl_mix = AverageMeter()
                if opt.positive_method == "cutmix" or opt.positive_method == "snapmix":
                    mixed_positive_samples1, mixed_positive_samples2, lam = vanilla_cutmix(images1, images2, opt)
                elif opt.positive_method == "saliencymix":
                    mixed_positive_samples1, mixed_positive_samples2, lam = salient_cutmix(images1, images2, model,
                                                                                           opt)
                elif opt.positive_method == "layersaliencymix":
                    mixed_positive_samples1, mixed_positive_samples2, lam = salient_cutmix(images1, images2,
                                                                                                 model, opt)
                elif opt.positive_method == "attentive_mix":
                    mixed_positive_samples1, mixed_positive_samples2, lam = attentive_cutmix(images1, images2, opt)
                elif opt.positive_method == "cv2saliency":
                    mixed_positive_samples1, mixed_positive_samples2, lam = salient_cutmix(images1, images2, model,
                                                                                           opt)
                else:
                    mixed_positive_samples1, mixed_positive_samples2, labels_new, lam = vanilla_mixup(images1,
                                                                                                      images2,
                                                                                                      labels,
                                                                                                      alpha=opt.alpha_vanilla,
                                                                                                      beta=opt.beta_vanilla,
                                                                                                      mode=opt.positive_method,
                                                                                                      encoder=model)

                mixed_positive_samples = torch.cat([mixed_positive_samples1, mixed_positive_samples2], dim=0)
                mixed_positive_features = model(mixed_positive_samples)
                mixed_positive_features1, mixed_positive_features2 = torch.split(mixed_positive_features,
                                                                                 [bsz, bsz],
                                                                                 dim=0)
                mixed_positive_features = torch.cat(
                    [mixed_positive_features1.unsqueeze(1), mixed_positive_features2.unsqueeze(1)], dim=1)
                loss_ssl_mix = criterion1(features,
                                          features_positive=mixed_positive_features)  # no criterion1(mixed_positive_features) since the patched areas are still the same
                if opt.old_augmented:
                    loss_ssl = loss_ssl + lam * loss_ssl_mix
                else:
                    loss_ssl = loss2 = loss_ssl_mix
                losses_ssl_mix.update(loss_ssl_mix.detach().cpu().item())

            loss = opt.method_gama * loss_sup + opt.method_lam * loss_ssl
            losses_ssl.update(loss_ssl.detach().cpu().item())
            losses_sup.update(loss_sup.detach().cpu().item())
            #ious_epoch.append(ious)

        elif opt.method == 'SimCLR_CE':
            features = model(images)
            features1, features2 = torch.split(features, [bsz, bsz], dim=0)
            features = torch.cat([features1.unsqueeze(1), features2.unsqueeze(1)], dim=1)
            loss_ce = loss1 = criterion2(logits, labels_linear)
            loss_ssl = loss2 = criterion1(features)
            loss = opt.method_gama * loss_ce + opt.method_lam * loss_ssl

            losses_ssl.update(loss_ssl.detach().cpu().item())
            losses_sup.update(loss_ce.detach().cpu().item())

        elif opt.method == "MoCo":
            logits, labels_moco = model(images1, images2, mode="moco")
            loss_moco = loss2 = criterion1(logits, labels_moco)

            if opt.mixup_positive:

                if opt.positive_method == "cutmix":
                    mixed_positive_samples1, mixed_positive_samples2, lam = vanilla_cutmix(images1, images2, opt)
                elif opt.positive_method == "saliencymix":
                    mixed_positive_samples1, mixed_positive_samples2, lam = salient_cutmix(images1, images2, model,
                                                                                           opt)
                elif opt.positive_method == "layersaliencymix":
                    mixed_positive_samples1, mixed_positive_samples2, lam = salient_cutmix(images1, images2, model,
                                                                                           opt)
                else:
                    mixed_positive_samples1, mixed_positive_samples2, labels_new, lam = vanilla_mixup(images1,
                                                                                                      images2,
                                                                                                      labels,
                                                                                                      alpha=opt.alpha_vanilla,
                                                                                                      beta=opt.beta_vanilla,
                                                                                                      mode=opt.positive_method,
                                                                                                      encoder=model)
                logits_mix, labels_moco_mix = model(mixed_positive_samples1, mixed_positive_samples2, mode="moco")
                loss_moco_mix = criterion1(logits_mix, labels_moco_mix)
                loss_moco = loss2 = loss_moco + lam * loss_moco_mix

            loss1 = loss2
            loss = opt.method_lam * loss_moco  # opt.method_gama * loss_sup +

        else:
            raise ValueError('contrastive method not supported: {}'.
                             format(opt.method))

        # update metric
        losses.update(loss.detach().cpu().item(), bsz)
        losses1.update(loss1.detach().cpu().item(), bsz)
        losses2.update(loss2.detach().cpu().item(), bsz)

        # print info
        if (idx + 1) % 100 == 0:
            print('Train: [{0}][{1}/{2}]\t'
                  'loss_vali {loss.val:.3f} ({loss.avg:.3f})\t'
                  "loss1_vali {loss1.val:.3f} ({loss1.avg:.3f})\t"
                  "loss2_vali {loss2.val:.3f} ({loss2.avg:.3f})\t".format(
                epoch, idx + 1, len(vali_loader), loss=losses, loss1=losses1, loss2=losses2))
            if opt.method == "SimCLR":
                print("loss_ssl_vali {loss_ssl.val:.3f} ({loss_ssl.avg:.3f})\t"
                      "loss_sup_vali {loss_sup.val:.3f} ({loss_sup.avg:.3f})\t".format(
                    loss_ssl=losses_ssl, loss_sup=losses_sup))
                if opt.mixup_positive:
                    print("loss_ssl_mix_vali {loss_ssl_mix.val:.3f} ({loss_ssl_mix.avg:.3f})\t".format(
                        loss_ssl_mix=losses_ssl_mix))
            if opt.method == "SimCLR_CE":
                print("loss_ssl_vali {loss_ssl.val:.3f} ({loss_ssl.avg:.3f})\t"
                      "loss_sup_vali {loss_ce.val:.3f} ({loss_ce.avg:.3f})\t".format(
                    loss_ssl=losses_ssl, loss_ce=losses_sup))
            sys.stdout.flush()

    return (losses.avg, losses1.avg, losses2.avg), ious_epoch


def main():
    opt = parse_option()

    # build data loader
    train_loader, test_loader = set_loader(opt)
    print("train_loader, ", train_loader.__len__())
    print("test_loader, ", test_loader.__len__())

    # build model and criterion
    model, linear, criterion1, criterion2 = set_model(opt)

    # build optimizer
    optimizer = set_optimizer(opt, model)

    losses = []
    all_ious = []
    all_ious_vali = []
    losses_vali = []

    # training routine
    for epoch in range(0, opt.epochs):
        adjust_learning_rate(opt, optimizer, epoch)

        # train for one epoch
        time1 = time.time()
        loss, ious_epoch, grads = train(train_loader, model, linear, criterion1, criterion2, optimizer, epoch, opt)
        #loss_vali, ious_epoch_vali = validate(test_loader, model, linear, criterion1, criterion2, optimizer, epoch, opt)
        time2 = time.time()
        print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))


        losses.append(loss)
        all_ious.append(ious_epoch)
        # losses_vali.append(loss_vali)
        # all_ious_vali.append(ious_epoch_vali)

        if epoch % opt.save_freq == 0:
            save_file = os.path.join(
                opt.save_folder, 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
            save_model(model, linear, optimizer, opt, epoch, save_file)

        """
        with open(os.path.join(opt.save_folder, "iou_" + str(epoch)), "wb") as f:
            pickle.dump(ious_epoch, f)
        #with open(os.path.join(opt.save_folder, "iou_vali_" + str(epoch)), "wb") as f:
        #    pickle.dump(ious_epoch_vali, f)
        with open(os.path.join(opt.save_folder, "grad_" + str(epoch)), "wb") as f:
            pickle.dump(grads, f)
        #with open(os.path.join(opt.save_folder, "hessian_" + str(epoch)), "wb") as f:
        #    pickle.dump(hessian, f)
        with open(os.path.join(opt.save_folder, "loss_" + str(epoch)), "wb") as f:
            pickle.dump(loss, f)
        """

    # save the last model
    save_file = os.path.join(
        opt.save_folder, 'last.pth')
    save_model(model, linear, optimizer, opt, opt.epochs, save_file)
    with open(os.path.join(opt.save_folder, "loss_" + str(opt.trail)), "wb") as f:
        pickle.dump((losses, losses_vali), f)
    with open(os.path.join(opt.save_folder, "iou_all"), "wb") as f:
        pickle.dump(all_ious, f)
    with open(os.path.join(opt.save_folder, "iou_all_vali"), "wb") as f:
        pickle.dump(all_ious_vali, f)


if __name__ == '__main__':
    main()