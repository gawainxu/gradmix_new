from __future__ import print_function

import os
import sys
import argparse
import time
import math
import pickle
import numpy as np

import torch
import torch.backends.cudnn as cudnn
from torchvision import datasets

from util import TwoCropTransform, AverageMeter
from util import adjust_learning_rate
from util import set_optimizer, save_model
from datautil import num_inlier_classes_mapping
from networks.resnet_multi import SupConResNet_end, SupConResNet_inter
from losses import SupConLoss
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

    parser.add_argument('--print_freq', type=int, default=10,
                        help='print frequency')
    parser.add_argument('--save_freq', type=int, default=50,
                        help='save frequency')
    parser.add_argument('--batch_size', type=int, default=128,
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
    parser.add_argument('--model', type=str, default='resnet18',
                        choices=["resnet18", "resnet34", "resnet50", "preactresnet18", "preactresnet34"])
    parser.add_argument("--feat_dim", type=int, default=128)
    parser.add_argument('--datasets', type=str, default='cifar10', help='dataset')
    parser.add_argument('--mean', type=str, help='mean of dataset in path in form of str tuple')
    parser.add_argument('--std', type=str, help='std of dataset in path in form of str tuple')
    parser.add_argument('--data_folder', type=str, default=None, help='path to custom dataset')
    parser.add_argument('--size', type=int, default=32, help='parameter for RandomResizedCrop')
    parser.add_argument("--augmix", type=bool, default=False)

    # method
    parser.add_argument('--method', type=str, default='SimCLR',
                        choices=['SupCon', 'SimCLR', "SimCLR_CE", "MoCo"], help='choose method')
    parser.add_argument("--ensemble_mode", type=str, default="end")
    parser.add_argument("--trail", type=int, default=0, choices=[0, 1, 2, 3, 4, 5, 6],
                        help="index of repeating training")
    parser.add_argument("--action", type=str, default="training_supcon",
                        choices=["training_supcon", "trainging_linear", "testing_known", "testing_unknown",
                                 "feature_reading"])
    # temperature
    parser.add_argument('--temp1', type=float, default=0.05, help='temperature for loss1')
    parser.add_argument('--temp2', type=float, default=0.05, help='temperature for loss2')
    parser.add_argument('--temp3', type=float, default=0.05, help='temperature for loss3')

    parser.add_argument('--alpha1', type=float, default=1, help='coefficient for loss1')
    parser.add_argument('--alpha2', type=float, default=1, help='coefficient for loss2')
    parser.add_argument('--alpha3', type=float, default=1, help='coefficient for loss3')

    # other setting
    parser.add_argument('--cosine', type=bool, default=False,
                        help='using cosine annealing')
    parser.add_argument('--syncBN', action='store_true',
                        help='using synchronized batch normalization')
    parser.add_argument('--warm', action='store_true',
                        help='warm-up for large batch training')
    parser.add_argument("--data_method", type=str, default="original",
                        choices=["original", "upsampling"])

    # upsampling parameters for feature boundary based augmentation
    parser.add_argument("--upsample", type=bool, default=False)
    parser.add_argument("--portion_out", type=float, default=0.5)
    parser.add_argument("--upsample_times", type=int, default=1)
    parser.add_argument("--last_feature_path", type=str, default=None)
    parser.add_argument("--last_model_path", type=str, default=None)

    # mixup parameters
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
    opt.model_name += "_ensemble_"

    opt.model_name += 'trail_{}'.format(opt.trail) + "_" + str(opt.feat_dim) + "_" + str(opt.batch_size) + "_" + str(opt.ensemble_mode)
    opt.model_name += "_" + str(opt.temp1)
    opt.model_name += "_" + str(opt.temp2)
    opt.model_name += "_" + str(opt.temp3)

    opt.model_name += "_" + str(opt.alpha1)
    opt.model_name += "_" + str(opt.alpha2)
    opt.model_name += "_" + str(opt.alpha3)

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

    return opt


def set_loader(opt):
    # construct data loader

    train_dataset = get_train_datasets(opt)
    test_dataset = get_test_datasets(opt)

    train_sampler = None
    if opt.datasets != "imagenet100":
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True,
                                                   num_workers=opt.num_workers, pin_memory=True, sampler=train_sampler,
                                                   drop_last=True,
                                                   persistent_workers=True)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1,
                                                  shuffle=False,
                                                  num_workers=opt.num_workers, pin_memory=True, sampler=train_sampler,
                                                  drop_last=True,
                                                  persistent_workers=True)
    else:
        train_loader = train_dataset
        test_loader = test_dataset

    return train_loader, test_loader


def set_model(opt):
    if opt.datasets == "mnist":
        in_channels = 1
    else:
        in_channels = 3

    if opt.ensemble_mode == "end":
        model = SupConResNet_end(name=opt.model, feat_dim=opt.feat_dim, in_channels=in_channels)
    else:
        model = SupConResNet_inter(name=opt.model, feat_dim=opt.feat_dim, in_channels=in_channels)

    criterion1 = SupConLoss(temperature=opt.temp1)
    criterion2 = SupConLoss(temperature=opt.temp2)
    criterion3 = SupConLoss(temperature=opt.temp3)

    if opt.last_model_path is not None:
        print("model loaded")
        load_model(opt, model)

    # enable synchronized Batch Normalization
    if opt.syncBN:
        model = apex.parallel.convert_syncbn_model(model)

    if torch.cuda.is_available() and opt.use_cuda is True:

        if torch.cuda.device_count() > 1:
            model.encoder = torch.nn.DataParallel(model.encoder)

        model = model.cuda()
        criterion1 = criterion1.cuda()
        criterion2 = criterion2.cuda()
        criterion3 = criterion3.cuda()
        cudnn.benchmark = True

    return model, criterion1, criterion2, criterion3



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


def train(train_loader, model, criterion1, criterion2, criterion3, optimizer, epoch, opt):
    """one epoch training"""
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    losses1 = AverageMeter()
    losses2 = AverageMeter()
    losses3 = AverageMeter()

    end = time.time()

    for idx, (images, labels, annotations) in enumerate(train_loader):

        data_time.update(time.time() - end)
        images1 = images[0]
        images2 = images[1]

        if torch.cuda.is_available():
            images1 = images1.cuda(non_blocking=True)
            images2 = images2.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)

        images = torch.cat([images1, images2], dim=0)

        bsz = labels.shape[0]

        # warm-up learning rate
        # warmup_learning_rate(opt, epoch, idx, len(train_loader), optimizer)

        features1, features2, features3 = model(images)
        features11, features12 = torch.split(features1, [bsz, bsz], dim=0)
        features1 = torch.cat([features11.unsqueeze(1), features12.unsqueeze(1)], dim=1)

        features21, features22 = torch.split(features2, [bsz, bsz], dim=0)
        features2 = torch.cat([features21.unsqueeze(1), features22.unsqueeze(1)], dim=1)

        features31, features32 = torch.split(features3, [bsz, bsz], dim=0)
        features3 = torch.cat([features31.unsqueeze(1), features32.unsqueeze(1)], dim=1)

        if opt.ensemble_mode == "end":
            loss1 = criterion1(features1, labels)
            loss2 = criterion2(features2, labels)
            loss3 = criterion3(features3, labels)
            loss = opt.alpha1 * loss1 + opt.alpha2 * loss2 + opt.alpha3 * loss3

        # update metric
        losses.update(loss.detach().cpu().item(), bsz)
        losses1.update(loss1.detach().cpu().item(), bsz)
        losses2.update(loss2.detach().cpu().item(), bsz)
        losses3.update(loss3.detach().cpu().item(), bsz)

        # SGD
        optimizer.zero_grad()
        loss.backward(retain_graph=False)
        # plot_grad_flow(model.named_parameters(), idx, epoch)
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
                  "loss2 {loss2.val:.3f} ({loss2.avg:.3f})\t"
                  "loss3 {loss3.val:.3f} ({loss3.avg:.3f})\t".format(
                epoch, idx + 1, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, loss1=losses1, loss2=losses2, loss3=losses3))
            sys.stdout.flush()

    return losses.avg, losses1.avg, losses2.avg, losses3.avg


def validate(vali_loader, model, criterion1, criterion2, criterion3, epoch, opt):
    """one epoch training"""

    model.eval()
    losses = AverageMeter()
    losses1 = AverageMeter()
    losses2 = AverageMeter()
    losses3 = AverageMeter()

    for idx, (images, labels, annotations) in enumerate(vali_loader):

        images1 = images[0]
        images2 = images[1]

        if torch.cuda.is_available():
            images1 = images1.cuda(non_blocking=True)
            images2 = images2.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
        images = torch.cat([images1, images2], dim=0)

        bsz = labels.shape[0]

        features = model(images)
        features1, features2 = torch.split(features, [bsz, bsz], dim=0)
        features = torch.cat([features1.unsqueeze(1), features2.unsqueeze(1)], dim=1)
        loss1 = criterion1(features, labels)
        loss2 = criterion2(features, labels)
        loss3 = criterion3(features, labels)
        loss =  opt.alpha1 * loss1 + opt.alpha2 * loss2 + opt.alpha3 * loss3

        # update metric
        losses.update(loss.detach().cpu().item(), bsz)
        losses1.update(loss1.detach().cpu().item(), bsz)
        losses2.update(loss2.detach().cpu().item(), bsz)
        losses3.update(loss3.detach().cpu().item(), bsz)

        # print info
        if (idx + 1) % 100 == 0:
            print('Train: [{0}][{1}/{2}]\t'
                  'loss_vali {loss.val:.3f} ({loss.avg:.3f})\t'
                  "loss1_vali {loss1.val:.3f} ({loss1.avg:.3f})\t"
                  "loss2_vali {loss2.val:.3f} ({loss2.avg:.3f})\t"
                  "loss3_vali {loss3.val:.3f} ({loss3.avg:.3f})\t".format(
                epoch, idx + 1, len(vali_loader), loss=losses, loss1=losses1, loss2=losses2, loss3=losses3))
            sys.stdout.flush()

    return losses.avg, losses1.avg, losses2.avg, losses3.avg


def main():
    opt = parse_option()

    # build data loader
    train_loader, test_loader = set_loader(opt)
    print("train_loader, ", train_loader.__len__())
    print("test_loader, ", test_loader.__len__())

    # build model and criterion
    model, criterion1, criterion2, criterion3 = set_model(opt)

    # build optimizer
    optimizer = set_optimizer(opt, model)

    losses = []
    losses1 = []
    losses2 = []
    losses3 = []
    losses_vali = []

    # training routine
    for epoch in range(0, opt.epochs):
        adjust_learning_rate(opt, optimizer, epoch)

        # train for one epoch
        time1 = time.time()
        loss, loss1, loss2, loss3 = train(train_loader, model, criterion1, criterion2, criterion3, optimizer, epoch, opt)
        # loss_vali, ious_epoch_vali = validate(test_loader, model, linear, criterion1, criterion2, optimizer, epoch, opt)
        time2 = time.time()
        print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))

        losses.append(loss)
        losses1.append(loss1)
        losses2.append(loss2)
        losses3.append(loss3)
        # losses_vali.append(loss_vali)

        if epoch % opt.save_freq == 0:
            save_file = os.path.join(
                opt.save_folder, 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
            save_model(model, None, optimizer, opt, epoch, save_file)

    # save the last model
    save_file = os.path.join(
        opt.save_folder, 'last.pth')
    save_model(model, None, optimizer, opt, opt.epochs, save_file)
    with open(os.path.join(opt.save_folder, "loss_" + str(opt.trail)), "wb") as f:
        pickle.dump((losses, losses1, losses2, losses3), f)


if __name__ == '__main__':
    main()
