from __future__ import print_function

import os
import sys
import argparse
import math
import pickle
import time

import torch
import torch.backends.cudnn as cudnn

from util import AverageMeter
from util import adjust_learning_rate, warmup_learning_rate
from util import set_optimizer, save_model
from datautil import get_train_datasets, get_test_datasets
from networks.resnet_big import SupConResNet, SupConResNet_MultiHead_remix
from networks.resnet_multi import SupConResNet_MultiHead
from networks.ts_models import basic_CNN_multihead
from losses import SupConLoss

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

    # model dataset
    parser.add_argument('--model', type=str, default='basic_cnn_multi')
    parser.add_argument("--last_model_path", type=str, default=None)
    parser.add_argument('--datasets', type=str, default='paderborn', help='dataset')
    parser.add_argument('--data_folder', type=str, default=None, help='path to custom dataset')
    parser.add_argument("--num_channels", type=int, default=1)

    # method
    parser.add_argument('--method', type=str, default='SupCon', help='choose method')
    parser.add_argument("--trail", type=int, default=0, choices=[0, 1, 2, 3, 4, 5], help="index of repeating training")
    parser.add_argument("--action", type=str, default="training_supcon",
                        choices=["training_supcon", "trainging_linear", "testing_known", "testing_unknown",
                                 "feature_reading"])
    # temperature
    parser.add_argument('--temp1', type=float, default=0.005, help='temperature for loss')
    parser.add_argument('--temp2', type=float, default=0.01, help='temperature for loss')
    parser.add_argument('--temp3', type=float, default=0.05, help='temperature for loss')
    parser.add_argument("--out_dim", type=int, default=512, help="output dimension of the resnet blocks")
    parser.add_argument("--feat_dim", type=int, default=128)

    # other setting
    parser.add_argument('--cosine', type=bool, default=False,
                        help='using cosine annealing')
    parser.add_argument('--syncBN', action='store_true',
                        help='using synchronized batch normalization')
    parser.add_argument('--warm', action='store_true',
                        help='warm-up for large batch training')

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

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    opt.model_name = opt.datasets + "_" + opt.model + '_trail_{}'.format(opt.trail) + "_" + str(
            opt.feat_dim) + "_" + str(opt.out_dim) + "_" + str(opt.temp1) + "_" + str(opt.temp2) + "_" + str(
            opt.temp3) + "_" + str(opt.batch_size)

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

    return opt


def set_loader(opt):

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

    if opt.model == "basic_cnn_multi":
        model = basic_CNN_multihead(feature_dim=opt.feat_dim, in_channels=opt.num_channels)
    elif opt.model == "resnet_multi":
        if torch.cuda.device_count() > 1:
            model = SupConResNet_MultiHead_remix(output_dim=opt.out_dim, feat_dim=opt.feat_dim, in_channels=opt.num_channels)
        else:
            model = SupConResNet_MultiHead(output_dim=opt.out_dim, feat_dim=opt.feat_dim, in_channels=opt.num_channels)

    if opt.last_model_path is not None:
        model = load_model(opt, model)

    # enable synchronized Batch Normalization
    if opt.syncBN:
        model = apex.parallel.convert_syncbn_model(model)

    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model)
        model = model.cuda()
        cudnn.benchmark = True

    criterion1 = SupConLoss(temperature=opt.temp1)
    criterion2 = SupConLoss(temperature=opt.temp2)
    criterion3 = SupConLoss(temperature=opt.temp3)
    criterion1 = criterion1.cuda()
    criterion2 = criterion2.cuda()
    criterion3 = criterion3.cuda()

    return model, (criterion1, criterion2, criterion3)


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


def train(train_loader, model, criterions, optimizer, epoch, opt):
    """one epoch training"""
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    losses1 = AverageMeter()
    losses2 = AverageMeter()
    losses3 = AverageMeter()

    criterion1, criterion2, criterion3 = criterions

    end = time.time()
    for idx, (images, labels) in enumerate(train_loader):

        data_time.update(time.time() - end)
        images1 = images[0]
        images2 = images[1]

        images = torch.cat([images1, images2], dim=0)

        if torch.cuda.is_available():
            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)

        bsz = labels.shape[0]
        # warm-up learning rate
        # warmup_learning_rate(opt, epoch, idx, len(train_loader), optimizer)

        # compute loss
        features1, features2, features3 = model(images)

        features1_1, features1_2 = torch.split(features1, [bsz, bsz], dim=0)
        features2_1, features2_2 = torch.split(features2, [bsz, bsz], dim=0)
        features3_1, features3_2 = torch.split(features3, [bsz, bsz], dim=0)
        features1 = torch.cat([features1_1.unsqueeze(1), features1_2.unsqueeze(1)], dim=1)
        features2 = torch.cat([features2_1.unsqueeze(1), features2_2.unsqueeze(1)], dim=1)
        features3 = torch.cat([features3_1.unsqueeze(1), features3_2.unsqueeze(1)], dim=1)

        loss1 = criterion1(features1, labels)
        loss2 = criterion2(features2, labels)
        loss3 = criterion3(features3, labels)
        loss = loss1 + loss2 + loss3
        losses1.update(loss1.item(), bsz)
        losses2.update(loss2.item(), bsz)
        losses3.update(loss3.item(), bsz)

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
                  'loss {loss.val:.3f} ({loss.avg:.3f})\t'
                  'loss1 {loss1.val:.3f} ({loss1.avg:.3f})\t'
                  'loss2 {loss2.val:.3f} ({loss2.avg:.3f})\t'
                  'loss3 {loss3.val:.3f} ({loss3.avg:.3f})\t'.format(
                epoch, idx + 1, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, loss1=losses1, loss2=losses2, loss3=losses3))
            sys.stdout.flush()

    return losses.avg, losses1.avg, losses2.avg, losses3.avg


def main():
    opt = parse_option()

    # build data loader
    train_loader = set_loader(opt)

    # build model and criterion
    model, criterions = set_model(opt)

    # build optimizer
    optimizer = set_optimizer(opt, model)

    losses = []
    losses1 = []
    losses2 = []
    losses3 = []

    # training routine
    for epoch in range(0, opt.epochs):
        adjust_learning_rate(opt, optimizer, epoch)

        # train for one epoch
        time1 = time.time()
        loss, loss1, loss2, loss3 = train(train_loader, model, criterions, optimizer, epoch, opt)
        time2 = time.time()
        print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))

        losses.append(loss)
        losses1.append(loss1)
        losses2.append(loss2)
        losses3.append(loss3)
        if epoch % opt.save_freq == 0:
            save_file = os.path.join(
                opt.save_folder, 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
            save_model(model, optimizer, opt, epoch, save_file)

    # save the last model
    save_file = os.path.join(
        opt.save_folder, 'last.pth')
    save_model(model, optimizer, opt, opt.epochs, save_file)
    with open(os.path.join(opt.save_folder, "loss_" + str(opt.trail)), "wb") as f:
        pickle.dump((losses, losses1, losses2, losses3), f)


if __name__ == '__main__':
    main()
