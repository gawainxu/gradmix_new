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
import pickle

import torch
import torch.backends.cudnn as cudnn

from util import AverageMeter
from util import adjust_learning_rate, accuracy
from util import set_optimizer, save_model
from datautil import num_inlier_classes_mapping
from networks.ts_models import basic_CNN
from datautil import get_train_datasets, get_test_datasets

import matplotlib
matplotlib.use('Agg')

try:
    import apex
    from apex import amp, optimizers
except ImportError:
    pass


def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--print_freq', type=int, default=100,
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
    parser.add_argument("--action", type=str, default="training_ce",
                        choices=["training_supcon", "training_ce", "trainging_linear", "testing_known", "testing_unknown", "feature_reading"])        

    # model dataset
    parser.add_argument('--model', type=str, default='basic_cnn')
    parser.add_argument("--last_model_path", type=str, default=None)
    parser.add_argument('--datasets', type=str, default='paderborn')
    parser.add_argument('--data_folder', type=str, default=None)
    parser.add_argument("--num_channels", type=int, default=1)
    parser.add_argument("--split_mode", type=str, default='window')
    parser.add_argument('--win_len', type=int, default=18256)
    parser.add_argument('--split_step', type=int, default=100)
    parser.add_argument("--seperate_conditions", type=bool, default=False)
    parser.add_argument("--condition_name", type=str, default="")
    parser.add_argument("--data_channel", type=str, default="vibration")
    parser.add_argument("--trail", type=int, default=0)
    parser.add_argument("--crop_length", type=int, default=4096)
    parser.add_argument("--view_mode", type=str, default="scale_noise")
    parser.add_argument("--feat_dim", type=int, default=128)

        
    # other setting
    parser.add_argument('--cosine', type=bool, default=False,
                        help='using cosine annealing')
    parser.add_argument('--syncBN', action='store_true',
                        help='using synchronized batch normalization')
    parser.add_argument('--warm', action='store_true',
                        help='warm-up for large batch training')
    parser.add_argument("--use_cuda", type=bool, default=True)


    opt = parser.parse_args()

    # set the path according to the environment
    if opt.data_folder is None:
        opt.data_folder = '../datasets/'
    opt.model_path = './save/CE/{}_models'.format(opt.datasets)

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    opt.model_name = opt.datasets + "_" + opt.model
   
    opt.model_name += 'trail_{}'.format(opt.trail) + "_" + str(opt.feat_dim) + "_" + str(opt.batch_size)

    opt.save_folder = os.path.join(opt.model_path, opt.model_name)
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)
    
    opt.num_classes = num_inlier_classes_mapping[opt.datasets]

    return opt


def set_loader(opt):
    # construct data loader

    train_dataset = get_train_datasets(opt)
    test_dataset = get_test_datasets(opt)

    train_sampler = None
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True,
                                                num_workers=opt.num_workers, pin_memory=True, sampler=train_sampler, drop_last=True,
                                                persistent_workers = False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1,
                                              shuffle=False, num_workers=opt.num_workers, pin_memory=True,
                                              sampler=train_sampler, drop_last=True, persistent_workers=False)

    return train_loader, test_loader


def set_model(opt):
    if opt.model == "basic_cnn":
        model = basic_CNN(feature_dim=opt.feat_dim, in_channels=opt.num_channels,
                          num_classes= opt.num_classes)
        criterion = torch.nn.CrossEntropyLoss()

    if opt.last_model_path is not None:
        print("model loaded")
        load_model(opt, model)

    # enable synchronized Batch Normalization
    if opt.syncBN:
        model = apex.parallel.convert_syncbn_model(model)

    if torch.cuda.is_available() and opt.use_cuda is True:

        model = model.cuda()
        criterion = criterion.cuda()
        cudnn.benchmark = True

    return model, criterion



def load_model(opt, model=None):

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
    for idx, (images, labels) in enumerate(train_loader):

        data_time.update(time.time() - end)

        if torch.cuda.is_available():
            images = images.float().cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
       
        bsz = labels.shape[0]
        
        logits = model(images, return_embedding=False)
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


def test(test_loader, model, criterion, opt):
    """one epoch training"""
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    end = time.time()
    for idx, (images, labels) in enumerate(test_loader):

        data_time.update(time.time() - end)

        if torch.cuda.is_available():
            images = images.float().cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)

        bsz = labels.shape[0]

        logits = model(images, return_embedding=False)
        loss = criterion(logits, labels)
        losses.update(loss.item())
        acc = accuracy(logits, labels, topk=(1, 5))
        acc1, acc5 = acc[0].item(), acc[1].item()
        top1.update(acc1, bsz)
        top5.update(acc5, bsz)

        # update metric
        losses.update(loss.item(), bsz)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
    print('Test: \t'
              'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
              'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
              'Acc1 {top1.val:.3f} ({top1.avg:.3f})'
              'Acc5 {top5.val:.3f} ({top5.avg:.3f})'.format(
    batch_time=batch_time, loss=losses, top1=top1, top5=top5))

    return losses.avg, top1.avg, top5.avg


def main():
    opt = parse_option()

    # build data loader
    train_loader, test_loader = set_loader(opt)
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

        loss_test, acc1, acc5 = test(test_loader, model, criterion, opt)

        if epoch % opt.save_freq == 0:
            save_file = os.path.join(
                opt.save_folder, 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
            save_model(model, optimizer, opt, epoch, save_file)

    # save the last model
    save_file = os.path.join(
        opt.save_folder, 'last.pth')
    save_model(model, optimizer, opt, opt.epochs, save_file)
    with open(os.path.join(opt.save_folder, "loss_" + str(opt.trail)), "wb") as f:
         pickle.dump((losses, loss_test, acc1, acc5), f)


if __name__ == '__main__':
    main()
