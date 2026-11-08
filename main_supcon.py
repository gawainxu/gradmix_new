from __future__ import print_function

import os
import sys
import argparse
import time
import math
import pickle
import random
import copy

import torch
import torch.backends.cudnn as cudnn
from torchvision import transforms, datasets

from util import TwoCropTransform, AverageMeter
from util import adjust_learning_rate, warmup_learning_rate
from util import set_optimizer, save_model, label_convert
from datautil import vanilla_mixup, vanilla_cutmix, salient_cutmix
from datautil import num_inlier_classes_mapping, mixup_hybrid_features  #
from networks.resnet_big import SupConResNet, LinearClassifier
from networks.resnet_big import MoCoResNet
from networks.maskcon import MaskCon
from networks.simCNN import simCNN_contrastive
from networks.resnet_preact import SupConpPreactResNet
from networks.mlp import SupConMLP
from losses import SupConLoss
from loss_mixup import SupConLoss_mix
from datautil import get_train_datasets, mixup_negative  #
from gradient_cache import gradient_cache

import matplotlib
matplotlib.use('Agg')

try:
    import apex
    from apex import amp, optimizers
except ImportError:
    pass


def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--print_freq', type=int, default=50,
                        help='print frequency')
    parser.add_argument('--save_freq', type=int, default=100,
                        help='save frequency')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='batch_size')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of training epochs')

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
    parser.add_argument("--mixed_precision", type=bool, default=False)
    parser.add_argument("--grad_splits", type=int, default=1)

    # model dataset
    parser.add_argument('--model', type=str, default='resnet18', choices=["resnet18", "resnet34", "resnet50", "preactresnet18", "preactresnet34", "simCNN", "MLP", "MaskCon"])
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
                        choices=['SupCon', 'SimCLR', "SimCLR_CE", "MoCo", "MaskCon"], help='choose method')
    parser.add_argument("--method_gama", type=float, default=0.0)
    parser.add_argument("--method_lam", type=float, default=1.0)
    parser.add_argument("--method_w", type=float, default=0.5)
    parser.add_argument("--method_T1", type=float, default=0.05)
    parser.add_argument("--method_T2", type=float, default=0.1)
    parser.add_argument("--trail", type=int, default=0, choices=[0,1,2,3,4,5, 6], help="index of repeating training")
    parser.add_argument("--action", type=str, default="training_supcon",
                        choices=["training_supcon", "trainging_linear", "testing_known", "testing_unknown", "feature_reading"])            
        
    # temperature
    parser.add_argument('--temp', type=float, default=0.05, help='temperature for loss')
    parser.add_argument("--clip", type=float, default=None, help="for gradient clipping")

    # moco parameters
    parser.add_argument("--K", type=int, default=4096, help="buffer size in moco")
    parser.add_argument("--momentum_moco", type=float, default=0.999)

    # other setting
    parser.add_argument('--cosine', type=bool, default=False,
                        help='using cosine annealing')
    parser.add_argument('--syncBN', action='store_true',
                        help='using synchronized batch normalization')
    parser.add_argument('--warm', action='store_true',
                        help='warm-up for large batch training')
    parser.add_argument("--augmentation_method", type=str, default="vanilia", 
                        choices=["vanilia", "mixup_negative", "mixup_positive", "mixup_hybrid", "mixup_vanilla", "mixup_vanilla_features"])
    parser.add_argument("--mixup_supcon", type=str, default="no", choices=["no", "max_similarity"])
    parser.add_argument("--data_method", type=str, default="original", 
                        choices=["original", "upsampling"])
    parser.add_argument("--architecture", type=str, default="single", choices=["single", "multi"])
    parser.add_argument("--ensemble_num", type=int, default=1)
    parser.add_argument("--feat_dim", type=int, default=128)
    parser.add_argument("--old_augmented", type=bool, default=False)


    # upsampling parameters
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
    parser.add_argument("--positive_method", type=str, default="random", choices=["max_similarity", "min_similarity", "random", "prob_similarity", "reverse", "saliencymix", "layersaliencymix", "cutmix", "cv2saliency"])
    parser.add_argument("--grad_layers", type=str, default="3")
    parser.add_argument("--negative_method", type=str, default="no", choices=["max_similarity", "random", "even", "no"])
    parser.add_argument("--hybrid_method", type=str, default="no", choices=["min_similarity", "random", "no"])
    parser.add_argument("--vanilla_method", type=str, default="no", choices=["reverse", "random", "no"])
    parser.add_argument("--randaug", type=int, default=0)
    parser.add_argument("--apool", type=bool, default=False)
    parser.add_argument("--use_cuda", type=bool, default=True)


    opt = parser.parse_args()
    
    print("mixup_positive", opt.mixup_positive)

    # check if dataset is path that passed required arguments
    if opt.datasets == 'path':
        assert opt.data_folder is not None \
            and opt.mean is not None \
            and opt.std is not None

    # set the path according to the environment
    if opt.data_folder is None:
        opt.data_folder = '../datasets/'
    opt.model_path = './save/SupCon/{}_models'.format(opt.datasets)

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))
        

    opt.model_name = opt.datasets + "_" + opt.model
    if opt.data_method == "upsampling":
        opt.model_name += "_upsampling_data_" + str(opt.portion_out) + "_" + str(opt.upsample_times)
    else:
        opt.model_name += "_original_data_"

    if opt.augmentation_method == "mixup_negative":
        opt.model_name += "_mixup_negative_" + str(opt.negative_method) + "_intra_" + str(opt.intra_inter_mix_negative) + "_alpha_" + str(opt.alpha_negative)  
    elif opt.augmentation_method == "mixup_hybrid":
        opt.model_name += "_mixup_hybrid_" + str(opt.hybrid_method) + "_intra_" + str(opt.intra_inter_mix_hybrid) +  "_alpha_" + str(opt.alpha_hybrid) + "_beta_" + str(opt.beta_hybrid) + "_alfa_" + str(opt.alfa)             
    elif opt.augmentation_method == "mixup_positive":
        opt.model_name += "_mixup_positive_" + "alpha_" + str(opt.alpha_vanilla) + "_beta_" + str(opt.beta_vanilla) + "_" + opt.positive_method + "_" + opt.mixup_supcon + "_" + opt.grad_layers
    elif opt.augmentation_method == "mixup_vanilla_features":
        opt.model_name += "_mixup_vanilla_features_" + opt.vanilla_method + "_alpha_" + str(opt.alpha_vanilla) + "_beta_" + str(opt.beta_vanilla) + "_alfa_" + str(opt.alfa)
    elif opt.augmentation_method == "vanilia":
        opt.model_name += "_vanilia_"

    if opt.method == "SupCon":
        opt.model_name += "_SupCon_"
    elif opt.method == "SimCLR":
        opt.model_name += "_SimCLR_" + str(opt.method_gama) + "_" + str(opt.method_lam) + "_" + str(opt.temp) + "_"
    elif opt.method == "SimCLR_CE":
        opt.model_name += "_SimCLR_CE_" + str(opt.method_gama) + "_" + str(opt.method_lam) + "_"
    elif opt.method == "MaskCon":
        opt.model_name += "_MaskCon_" + str(opt.method_w) + "_" + str(opt.method_T1) + "_" + str(opt.method_T2) + "_"
    elif opt.method == "MoCo":
        opt.model_name += "_MoCo_" + str(opt.method_gama) + "_" +  str(opt.method_lam) + "_K_" + str(opt.K) + "_momentum_" + str(opt.momentum_moco) + "_"

    opt.model_name += 'trail_{}'.format(opt.trail) + "_" + str(opt.feat_dim) + "_" + str(opt.batch_size)

    if opt.last_model_path is not None:
        opt.model_name += "_twostage"

    if opt.randaug == 1:
        opt.model_name += "_randaug" + str(opt.argmentation_n) + "_" + str(opt.argmentation_m)
    
    if opt.augmix:
        opt.model_name += "_augmix"
    
    if opt.old_augmented:
        opt.model_name += "_old_augmented"

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

    print(opt.model_name)
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
        train_dataset =  get_train_datasets(opt, last_features_list=last_features_list, last_feature_labels_list=last_feature_labels_list, last_model=last_model)
    else:
        train_dataset =  get_train_datasets(opt)

    train_sampler = None
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=(train_sampler is None),
                                               num_workers=opt.num_workers, pin_memory=True, sampler=train_sampler, drop_last=True)

    return train_loader



def set_model(opt):

    if opt.datasets == "mnist":
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
    else:
        if opt.model in ["resnet18", "resnet34", "resnet50"]:
            model = SupConResNet(name=opt.model, feat_dim=opt.feat_dim, in_channels=in_channels)
        elif opt.model in ["preactresnet18", "preactresnet34"]:
            model = SupConpPreactResNet(name=opt.model, feat_dim=opt.feat_dim, in_channels=in_channels)
        elif opt.model == "MLP":
            model = SupConMLP(feat_dim=opt.feat_dim)
        else:
            model = simCNN_contrastive(opt, feature_dim=opt.feat_dim, in_channels=in_channels)
            
        criterion1 = SupConLoss(temperature=opt.temp)
        criterion2 = SupConLoss(temperature=opt.temp)
        linear = None
        
    if opt.last_model_path is not None:
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
    model.cuda()
    model.eval()

    return model



def train(train_loader, model, linear, criterion1, criterion2, optimizer, epoch, opt):
    """one epoch training"""
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    losses_ssl = AverageMeter()
    losses_sup = AverageMeter()
    losses1 = AverageMeter()
    losses2 = AverageMeter()

    end = time.time()
    
    if opt.mixed_precision is True:
        scaler = torch.cuda.amp.GradScaler()
    
    for idx, (images, labels) in enumerate(train_loader):

        data_time.update(time.time() - end)
        images1 = images[0]
        images2 = images[1]
        
        images = torch.cat([images1, images2], dim=0)

        if torch.cuda.is_available():
            images = images.cuda(non_blocking=True)
            images1 = images1.cuda(non_blocking=True)
            images2 = images2.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
       
        bsz = labels.shape[0]

        optimizer.zero_grad()

        if opt.grad_splits == 1:
            if opt.method == "MoCo":
            
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
                        mixed_positive_samples1, mixed_positive_samples2, labels_new, lam = vanilla_mixup(images1, images2, labels, alpha=opt.alpha_vanilla, 
                                                                                                          beta=opt.beta_vanilla, mode=opt.positive_method, encoder=model)   
                    logits_mix, labels_moco_mix = model(mixed_positive_samples1, mixed_positive_samples2, mode="moco")
                    loss_moco_mix = criterion1(logits_mix, labels_moco_mix)
                    
                    loss_moco += lam * loss_moco_mix
                    loss2 = loss_moco
                
                
                features = model(images, mode="simclr")
                features1, features2 = torch.split(features, [bsz, bsz], dim=0)
                features = torch.cat([features1.unsqueeze(1), features2.unsqueeze(1)], dim=1)
                loss_sup = loss1 = criterion2(features, labels)
                
                losses_ssl.update(loss_moco.item())
                losses_sup.update(loss_sup.item())
                loss = opt.method_gama * loss_sup + opt.method_lam * loss_moco

            elif opt.method == "MaskCon":
                loss, loss1, loss2 = model(im_k=images1, im_q=images2, coarse_label=labels, w=opt.method_w)                    # TODO Check augs

            elif opt.method == 'SupCon':
                features = model(images)
                features1, features2 = torch.split(features, [bsz, bsz], dim=0)
                features = torch.cat([features1.unsqueeze(1), features2.unsqueeze(1)], dim=1)
                loss_sup = loss1 = criterion1(features, labels)
                loss = loss2 = loss1
                losses_sup.update(loss_sup.item())
                    
            elif opt.method == 'SimCLR':
                
                features = model(images)
                features1, features2 = torch.split(features, [bsz, bsz], dim=0)
                features = torch.cat([features1.unsqueeze(1), features2.unsqueeze(1)], dim=1)
                if opt.mixup_negative:
                    mixed_negatve_samples1, mixed_negative_samples2 = mixup_negative(images1, images2, labels, labels, alpha=opt.alpha_negative)                               #mixed_negative_features1, mixed_negative_features2 = mixup_negative_features(features1, features2, labels, labels)
                    mixed_negative_samples = torch.cat([mixed_negatve_samples1, mixed_negative_samples2], dim=0)
                    mixed_negative_features = model(mixed_negative_samples)
                    mixed_negative_features1, mixed_negative_features2 = torch.split(mixed_negative_features, [bsz, bsz], dim=0)
                    mixed_negative_features = torch.cat([mixed_negative_features1.unsqueeze(1), mixed_negative_features2.unsqueeze(1)], dim=1)
                    if opt.mixed_precision is True:
                        with torch.autocast(device_type='cuda', dtype=torch.float16):
                            loss_sup = loss1 = criterion2(features=features, labels=labels)   
                            loss_ssl = loss2 = criterion1(features=features, features_negative=mixed_negative_features)
                    else:
                        loss_sup = loss1 = criterion2(features=features, labels=labels)   
                        loss_ssl = loss2 = criterion1(features=features, features_negative=mixed_negative_features)
                        
                if opt.mixup_positive:
                    if opt.positive_method == "cutmix":
                        mixed_positive_samples1, mixed_positive_samples2, lam = vanilla_cutmix(images1, images2, opt)
                    elif opt.positive_method == "saliencymix":
                        mixed_positive_samples1, mixed_positive_samples2, lam = salient_cutmix(images1, images2, model, opt)
                    elif opt.positive_method == "layersaliencymix":
                        mixed_positive_samples1, mixed_positive_samples2, lam = salient_cutmix(images1, images2, model, opt)
                    elif opt.positive_method == "cv2saliency":
                        mixed_positive_samples1, mixed_positive_samples2, lam = salient_cutmix(images1, images2, model, opt)
                    else:
                        mixed_positive_samples1, mixed_positive_samples2, labels_new, lam = vanilla_mixup(images1, images2, labels, alpha=opt.alpha_vanilla, 
                                                                                                          beta=opt.beta_vanilla, mode=opt.positive_method, encoder=model)                
                    mixed_positive_samples = torch.cat([mixed_positive_samples1, mixed_positive_samples2], dim=0)
                    mixed_positive_features = model(mixed_positive_samples)
                    mixed_positive_features1, mixed_positive_features2 = torch.split(mixed_positive_features, [bsz, bsz], dim=0)
                    mixed_positive_features = torch.cat([mixed_positive_features1.unsqueeze(1), mixed_positive_features2.unsqueeze(1)], dim=1)
                    if opt.mixup_supcon != "no":
                        if opt.mixed_precision is True:
                            with torch.autocast(device_type='cuda', dtype=torch.float16):
                                loss_sup1 = criterion2(features=features, labels=labels)
                                loss_sup2 = criterion2(features=features, features_positive=mixed_positive_features, labels=labels)
                                loss_sup = loss1 = loss_sup1 + lam * loss_sup2
                        else:
                            loss_sup1 = criterion2(features=features, labels=labels)
                            loss_sup2 = criterion2(features=features, features_positive=mixed_positive_features, labels=labels)
                            loss_sup = loss1 = loss_sup1 + lam * loss_sup2
                    else:
                        if opt.mixed_precision is True:
                            with torch.autocast(device_type='cuda', dtype=torch.float16):
                                loss_sup = loss1 = criterion2(features=features, labels=labels)
                        else:
                            loss_sup = loss1 = criterion2(features=features, labels=labels)
                            
                    if opt.mixed_precision is True:
                        with torch.autocast(device_type='cuda', dtype=torch.float16):
                            loss_ssl1 = criterion1(features)
                            loss_ssl2 = criterion1(features, features_positive=mixed_positive_features)
                            loss_ssl = loss2 = loss_ssl1 + lam * loss_ssl2
                    else:
                        loss_ssl1 = criterion1(features)
                        loss_ssl2 = criterion1(features, features_positive=mixed_positive_features)
                        loss_ssl = loss2 = loss_ssl1 #+ lam * loss_ssl2    # !!!!!!!!!!!!!!!!! temp hard-coded !!!!!!!!!!!!!!
                else:
                    if opt.mixed_precision is True:
                        with torch.autocast(device_type='cuda', dtype=torch.float16):
                            loss_sup = loss1 = criterion2(features, labels)                              
                            loss_ssl = loss2 = criterion1(features)
                    else:
                        loss_sup = loss1 = criterion2(features, labels)                              
                        loss_ssl = loss2 = criterion1(features)

                losses_ssl.update(loss_ssl.item())
                losses_sup.update(loss_sup.item())
                
                if opt.mixed_precision is True:
                    with torch.autocast(device_type='cuda', dtype=torch.float16):
                        loss = opt.method_gama * loss_sup + opt.method_lam * loss_ssl            
                else:
                    loss = opt.method_gama * loss_sup + opt.method_lam * loss_ssl      

            elif opt.method == 'SimCLR_CE':

                logits = linear(model.encoder(images))
                labels_linear = torch.cat([labels, labels], dim=0)
                loss_ce = loss1 = criterion2(logits, labels_linear)
                features = model(images)
                features1, features2 = torch.split(features, [bsz, bsz], dim=0)
                features = torch.cat([features1.unsqueeze(1), features2.unsqueeze(1)], dim=1)
                loss_ssl = loss2 = criterion1(features)

                losses_ssl.update(loss_ssl.item())
                losses_sup.update(loss_ce.item())
                loss = loss = opt.method_gama * loss_ce + opt.method_lam * loss_ssl

            else:
                raise ValueError('contrastive method not supported: {}'.
                                 format(opt.method))

        
            # update metric
            losses.update(loss.item(), bsz)
            losses1.update(loss1.item(), bsz)
            losses2.update(loss2.item(), bsz)
            # SGD
            loss.backward()
            if opt.clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), opt.clip)
        
        
        else: 
            scaler = torch.cuda.amp.GradScaler()
            
            if opt.mixup_positive:
               
                if opt.positive_method == "cutmix":
                   mixed_positive_samples1, mixed_positive_samples2, lam = vanilla_cutmix(images1, images2, opt)
                elif opt.positive_method == "layersaliencymix":
                    mixed_positive_samples1, mixed_positive_samples2, lam = salient_cutmix(images1, images2, model, opt)
                    optimizer.zero_grad()   # for removing gradients in gradmix
                else:
                   mixed_positive_samples1, mixed_positive_samples2, labels_new, lam = vanilla_mixup(images1, images2, labels, alpha=opt.alpha_vanilla, 
                                                                                                     beta=opt.beta_vanilla, mode=opt.positive_method, encoder=model)
                mixed_positive_samples = torch.cat([mixed_positive_samples1, mixed_positive_samples2], dim=0)
                gc2 = gradient_cache(model=model, splits=opt.grad_splits, fp16=False, loss_fcn=criterion1, grad_scalar=scaler, optimizer=optimizer, if_normal=True, lam=lam)
                loss = gc2(model_inputs=images, model_inputs_mix=mixed_positive_samples)
                print("loss", loss)
            else:
                gc2 = gradient_cache(model=model, splits=opt.grad_splits, fp16=False, loss_fcn=criterion1, grad_scalar=scaler, optimizer=optimizer, if_normal=True)
                loss = gc2(model_inputs=images)
                print("loss", loss)
            
                
            norms = 0
            for p in model.parameters():
                #print(p.grad.norm())
                norms += p.grad.norm()
            print("model gradients norms", norms)
            losses.update(loss.item(), bsz)

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
            if opt.method == "SimCLR_CE":
                print("loss_ssl {loss_ssl.val:.3f} ({loss_ssl.avg:.3f})\t"
                      "loss_sup {loss_ce.val:.3f} ({loss_ce.avg:.3f})\t".format(
                        loss_ssl=losses_ssl, loss_ce=losses_sup))
            sys.stdout.flush()

    return (losses.avg, losses1.avg, losses2.avg)


def main():

    """
    only for checking the gpus
    """
    for i in range(torch.cuda.device_count()):
        print("gpu", i, torch.cuda.get_device_properties(i).total_memory/1024**3)

    opt = parse_option()

    # build data loader
    train_loader = set_loader(opt)
    print("train_loader, ", train_loader.__len__())

    # build model and criterion
    model, linear, criterion1, criterion2 = set_model(opt)

    # build optimizer
    optimizer = set_optimizer(opt, model)

    losses = []

    #save_file = os.path.join(opt.save_folder, 'first.pth')
    #save_model(model, optimizer, opt, opt.epochs, save_file)

    # training routine
    for epoch in range(0, opt.epochs):
        adjust_learning_rate(opt, optimizer, epoch)

        # train for one epoch
        time1 = time.time()
        loss = train(train_loader, model, linear, criterion1, criterion2, optimizer, epoch, opt)
        time2 = time.time()
        print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))

        losses.append(loss)
        if epoch % opt.save_freq == 0:
            save_file = os.path.join(
                opt.save_folder, 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
            save_model(model, linear, optimizer, opt, epoch, save_file)

    # save the last model
    save_file = os.path.join(
        opt.save_folder, 'last.pth')
    save_model(model, linear, optimizer, opt, opt.epochs, save_file)
    with open(os.path.join(opt.save_folder, "loss_" + str(opt.trail)), "wb") as f:
         pickle.dump(losses, f)


if __name__ == '__main__':
    main()
    
