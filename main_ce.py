#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 20:38:04 2024

@author: zhi
"""
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
from util import set_optimizer, label_convert
from datautil import num_inlier_classes_mapping, mixup_hybrid_features
from networks.resnet_big import SupCEResNet
from losses import SupConLoss
from loss_mixup import SupConLoss_mix
from datautil import get_train_datasets, mixup_negative

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
    parser.add_argument('--save_freq', type=int, default=20,
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

    # model dataset
    parser.add_argument('--model', type=str, default='resnet18', choices=["resnet18", "resnet34", "preactresnet18", "preactresnet34", "simCNN", "MLP", "MaskCon"])
    parser.add_argument("--resnet_wide", type=int, default=1, help="factor for expanding channels in wide resnet")
    parser.add_argument('--datasets', type=str, default='cifar10',
                        choices=["cifar-10-100-10", "cifar-10-100-50", 'cifar10', "tinyimgnet", 'mnist', "svhn", "cub", "aircraft", "FUB"], help='dataset')
    parser.add_argument('--mean', type=str, help='mean of dataset in path in form of str tuple')
    parser.add_argument('--std', type=str, help='std of dataset in path in form of str tuple')
    parser.add_argument('--data_folder', type=str, default=None, help='path to custom dataset')
    parser.add_argument('--size', type=int, default=32, help='parameter for RandomResizedCrop')
    parser.add_argument("--augmentation_list", type=list, default=[])
    parser.add_argument("--argmentation_n", type=int, default=2)
    parser.add_argument("--argmentation_m", type=int, default=6)
    parser.add_argument("--randaug", type=int, default=0)

    # method
    parser.add_argument("--trail", type=int, default=0, choices=[0,1,2,3,4,5], help="index of repeating training")
    parser.add_argument("--action", type=str, default="training_ce",
                        choices=["training_supcon", "training_ce", "trainging_linear", "testing_known", "testing_unknown", "feature_reading"])            
        
    # other setting
    parser.add_argument('--cosine', type=bool, default=False,
                        help='using cosine annealing')
    parser.add_argument('--syncBN', action='store_true',
                        help='using synchronized batch normalization')
    parser.add_argument('--warm', action='store_true',
                        help='warm-up for large batch training')
    parser.add_argument("--feat_dim", type=int, default=128)
    


    opt = parser.parse_args()

    # check if dataset is path that passed required arguments
    if opt.datasets == 'path':
        assert opt.data_folder is not None \
            and opt.mean is not None \
            and opt.std is not None

    # set the path according to the environment
    if opt.data_folder is None:
        opt.data_folder = '../datasets/'
    opt.model_path = './save/CE/{}_models'.format(opt.datasets)

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    opt.model_name = opt.datasets + "_" + opt.model + "_" + str(opt.resnet_wide)
   
    opt.model_name += 'trail_{}'.format(opt.trail) + "_" + str(opt.feat_dim) + "_" + str(opt.batch_size)

    if opt.randaug == 1:
        opt.model_name += "_randaug" + str(opt.argmentation_n) + "_" + str(opt.argmentation_m)

    opt.save_folder = os.path.join(opt.model_path, opt.model_name)
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)
    
    opt.num_classes = num_inlier_classes_mapping[opt.datasets]

    return opt


def set_loader(opt):
    # construct data loader
    
    train_dataset =  get_train_datasets(opt)

    train_sampler = None
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=(train_sampler is None),
                                               num_workers=opt.num_workers, pin_memory=True, sampler=train_sampler, drop_last=True)

    return train_loader



def set_model(opt):

    if opt.datasets == "mnist":
        in_channels = 1
    elif opt.datasets == "FUB":
        in_channels = 1
    else:
        in_channels = 3
        
    model = SupCEResNet(name=opt.model, in_channels=in_channels, num_classes=opt.num_classes)                 # TODO
    criterion = torch.nn.CrossEntropyLoss()
       
    if torch.cuda.is_available():
        
        model = model.cuda()
        criterion = criterion.cuda()
        cudnn.benchmark = True

    return model, criterion



def load_model(opt, model=None):
    if model is None:
        model = SupCEResNet(name=opt.model, num_classes=opt.num_classes)

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



def save_model(model=None, optimizer=None, opt=None, epoch=0, save_file=None):
    print('==> Saving...')
    state = {'opt': opt,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,}
    torch.save(state, save_file)
    del state



def train(train_loader, model, criterion, optimizer, epoch, opt):
    """one epoch training"""
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    end = time.time()
    for idx, (images, labels, _) in enumerate(train_loader):

        data_time.update(time.time() - end)

        if torch.cuda.is_available():
            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
       
        bsz = labels.shape[0]
        
        logits = model(images)
        loss = criterion(logits, labels)
        losses.update(loss.item())


        # update metric
        losses.update(loss.item(), bsz)

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
                  'loss {loss.val:.3f} ({loss.avg:.3f})\t'.format(
                   epoch, idx + 1, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, ))
            sys.stdout.flush()

    return losses.avg


def main():
    opt = parse_option()

    # build data loader
    train_loader = set_loader(opt)
    print("train_loader, ", train_loader.__len__())

    # build model and criterion
    model, criterion= set_model(opt)

    # build optimizer
    optimizer = set_optimizer(opt, model)
    losses = []

    # training routine
    for epoch in range(0, opt.epochs):
        adjust_learning_rate(opt, optimizer, epoch)

        # train for one epoch
        time1 = time.time()
        loss = train(train_loader, model, criterion, optimizer, epoch, opt)
        time2 = time.time()
        print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))

        losses.append(loss)
        if epoch % opt.save_freq == 0:
            save_file = os.path.join(
                opt.save_folder, 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
            save_model(model, optimizer, opt, epoch, save_file)

    # save the last model
    save_file = os.path.join(
        opt.save_folder, 'last.pth')
    save_model(model, optimizer, opt, opt.epochs, save_file)
    with open(os.path.join(opt.save_folder, "loss_" + str(opt.trail)), "wb") as f:
         pickle.dump(losses, f)


if __name__ == '__main__':
    main()
