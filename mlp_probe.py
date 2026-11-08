"""
This file is used for evaluating
the backbone features using mlp
"""

import os
import sys
import time
import argparse
import torch
import torch.nn as nn

from main_linear import load_model
from util import adjust_learning_rate, warmup_learning_rate, accuracy
from datautil import get_train_datasets, get_test_datasets
from networks.resnet_big import SupConResNet
from networks.resnet_preact import SupConpPreactResNet
from networks.simCNN import simCNN_contrastive
from networks.mlp import SupConMLP
from util import set_optimizer, save_model
from util import AverageMeter


def parse_option():
    parser = argparse.ArgumentParser('argument for mlp probing')

    parser.add_argument('--model', type=str, default='resnet18', choices=["resnet18"])
    parser.add_argument("--feat_dim", type=int, default=128)
    parser.add_argument("--backbone_model_dir", type=str,
                        default="/home/zhi/projects/comprehensive_OSR/save/SupCon/tinyimgnet_models")
    parser.add_argument("--backbone_model_name", type=str, default="tinyimgnet_resnet18_vanilia__SimCLR_1.0_1.0_0.05_trail_5_128_256_old_augmented")
    parser.add_argument('--datasets', type=str, default='tinyimgnet',
                        choices=["cifar-10-100-10", "cifar-10-100-50", 'cifar10', "cifar100", "tinyimgnet",
                                 "imagenet100", "imagenet100_m", 'mnist', "svhn", "cub", "aircraft"], help='dataset')
    parser.add_argument("--trail", type=int, default=5, choices=[0, 1, 2, 3, 4, 5, 6],
                        help="index of repeating training")
    parser.add_argument("--num_classes", type=int, default=200)
    parser.add_argument("--randaug", type=int, default=0)
    parser.add_argument("--augmix", type=bool, default=False)

    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--action", type=str, default="trainging_linear",
                        choices=["training_supcon", "trainging_linear", "testing_known", "testing_unknown",
                                 "feature_reading"])
    parser.add_argument('--cosine', type=bool, default=False,
                        help='using cosine annealing')
    parser.add_argument('--lr_decay_epochs', type=str, default="100",
                        help='where to decay lr, can be a list')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='num of workers to use')
    parser.add_argument('--print_freq', type=int, default=100,
                        help='print frequency')

    opt = parser.parse_args()

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    opt.backbone_model_dir = os.path.join(opt.backbone_model_dir, opt.backbone_model_name)
    opt.backbone_model_path = os.path.join(opt.backbone_model_dir, "last.pth")

    return opt


class mlp(nn.Module):

    def __init__(self, in_dim, hidden_dim, num_classes):
        super(mlp, self).__init__()
        self.l1 = nn.Linear(in_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, num_classes)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm1d(hidden_dim)

    def forward(self, x):

        y = self.l1(x)
        y = self.bn(y)
        y = self.relu(y)
        y = self.l2(y)

        return y


def set_model(opt):

    criterion = nn.CrossEntropyLoss()
    classifier = mlp(in_dim=512, hidden_dim=opt.hidden_dim, num_classes=opt.num_classes)
    classifier = classifier.cuda()
    criterion = criterion.cuda()

    if opt.datasets == "mnist":
        in_channels = 1
    else:
        in_channels = 3

    if opt.model == "resnet18" or opt.model == "resnet34" or opt.model == "resnet50":
        model = SupConResNet(name=opt.model, feat_dim=opt.feat_dim, in_channels=in_channels)
    elif opt.model == "preactresnet18" or opt.model == "preactresnet34":
        model = SupConpPreactResNet(name=opt.model, feat_dim=opt.feat_dim, in_channels=in_channels)
    elif opt.model == "MLP":
        model = SupConMLP(feat_dim=opt.feat_dim)
    else:
        model = simCNN_contrastive(opt, feature_dim=opt.feat_dim, in_channels=in_channels)

    model = load_model(model, opt.backbone_model_path)

    return model, classifier, criterion


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


def train(train_loader, model, classifier, criterion, optimizer, epoch, opt):
    """one epoch training"""
    print("Training start")
    model.eval()
    classifier.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    end = time.time()
    for idx, (images, labels, _) in enumerate(train_loader):
        data_time.update(time.time() - end)

        images = images.cuda(non_blocking=True)
        labels = labels.cuda(non_blocking=True)
        bsz = labels.shape[0]

        # compute loss
        with torch.no_grad():
            features = model.encoder(images)
            features = features.cuda(non_blocking=True)

        output = classifier(features)
        loss = criterion(output, labels)

        # update metric
        losses.update(loss.item(), bsz)
        acc = accuracy(output, labels, topk=(1, 5))
        acc1, acc5 = acc[0].item(), acc[1].item()
        top1.update(acc1, bsz)
        top5.update(acc5, bsz)

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
                  'Acc1 {top1.val:.3f} ({top1.avg:.3f})'
                  'Acc5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                epoch, idx + 1, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1, top5=top5))
            sys.stdout.flush()

    return losses.avg, top1.avg, top5.avg


def validate(val_loader, model, classifier, criterion, opt):
    """validation"""
    model.eval()
    classifier.eval()

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    with torch.no_grad():
        end = time.time()
        for idx, (images, labels, _) in enumerate(val_loader):
            images = images.float().cuda()
            labels = labels.cuda()
            bsz = labels.shape[0]

            # forward
            output = classifier(model.encoder(images))
            loss = criterion(output, labels)

            # update metric
            losses.update(loss.item(), bsz)
            acc = accuracy(output, labels, topk=(1, 5))
            acc1, acc5 = acc[0].item(), acc[1].item()
            top1.update(acc1, bsz)
            top5.update(acc5, bsz)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if idx % (opt.print_freq * 10) == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc1 {top1.val:.3f} ({top1.avg:.3f})'
                      'Acc5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                    idx, len(val_loader), batch_time=batch_time,
                    loss=losses, top1=top1, top5=top5))

    return losses.avg, top1.avg, top5.avg


if __name__ == "__main__":

    opt = parse_option()
    # build model and criterion
    model, classifier, criterion = set_model(opt)
    optimizer = set_optimizer(opt, classifier)
    train_loader, test_loader = set_loader(opt)

    best_top1_test_acc = 0
    best_top5_test_acc = 0
    for epoch in range(opt.epochs):
        adjust_learning_rate(opt, optimizer, epoch)

        # train for one epoch
        time1 = time.time()
        loss, avg1_train, avg5_train = train(train_loader, model, classifier, criterion,
                                             optimizer, epoch, opt)
        time2 = time.time()
        print('Train epoch {}, total time {:.2f}, accuracy1:{:.2f}, accuracy5:{:.2f}, loss:{:.2f}'.format(
            epoch, time2 - time1, avg1_train, avg5_train, loss))

        loss_test, avg1_test, avg5_test = validate(test_loader, model, classifier, criterion, opt)
        print("test acc1", avg1_test)
        if avg1_test > best_top1_test_acc:
            best_top1_test_acc = avg1_test

        print("test acc5", avg5_test)
        if avg5_test > best_top5_test_acc:
            best_top5_test_acc = avg5_test

    save_file = opt.backbone_model_name.replace(".pth", "_mlp") + ".pth"
    save_file = os.path.join(opt.backbone_model_dir, save_file)
    save_model(model=model, linear=classifier, optimizer=optimizer, opt=opt, epoch=epoch, save_file=save_file)
    print("Best top1 accuacy is ", best_top1_test_acc)
    print("Best top5 accuacy is ", best_top5_test_acc)







