from __future__ import print_function

import os
import sys
import argparse
import time
import math
import copy

import torch
import torch.backends.cudnn as cudnn

from util import AverageMeter
from main_supcon_old import set_loader
from util import adjust_learning_rate, warmup_learning_rate, accuracy
from util import set_optimizer, save_model
from networks.resnet_big import SupConResNet, LinearClassifier, ResNet_Pretrain
from networks.resnet_big import MoCoResNet
from networks.vgg import SupConVGG
from networks.simCNN import simCNN_contrastive
from networks.mlp import SupConMLP
from datautil import osr_splits_inliers
from datautil import get_test_datasets

try:
    import apex
    from apex import amp, optimizers
except ImportError:
    pass


def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--print_freq', type=int, default=40,
                        help='print frequency')
    parser.add_argument("--action", type=str, default="trainging_linear",
                        choices=["training_supcon", "trainging_linear", 
                                 "testing_known", "testing_unknown", "feature_reading"])
    parser.add_argument('--save_freq', type=int, default=10,
                        help='save frequency')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='batch_size')
    parser.add_argument('--num_workers', type=int, default=16,
                        help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=20,
                        help='number of training epochs')

    # optimization 
    parser.add_argument('--learning_rate', type=float, default=0.1,
                        help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='60,75,90',
                        help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1,
                        help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.1,
                        help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum')

    # model dataset
    parser.add_argument('--method', type=str, default='SimCLR',
                        choices=['SupCon', 'SimCLR', "SimCLR_CE", "MoCo", "MaskCon"], help='choose method')
    parser.add_argument("--K", type=int, default=4096, help="buffer size in moco")
    parser.add_argument("--momentum_moco", type=float, default=0.9)
    parser.add_argument('--temp', type=float, default=0.05, help='temperature for loss')
    parser.add_argument("--clip", type=float, default=None, help="for gradient clipping")
    
    parser.add_argument('--model', type=str, default='vgg16', choices=["resnet18", "resnet34", "vgg16", "resnet50_pretrain", "simCNN", "MLP"])
    parser.add_argument("--feat_dim", type=int, default=128)
    parser.add_argument('--datasets', type=str, default='cifar10', help='dataset')
    parser.add_argument("--backbone_model_direct", type=str, default=None)
    parser.add_argument("--backbone_model_name", type=str, default="last.pth")
    parser.add_argument("--trail", type=int, default=0)
    parser.add_argument("--temp_list", type=str, default="")
    parser.add_argument("--randaug", type=int, default=0)
    parser.add_argument("--apool", type=bool, default=False)
    parser.add_argument("--augmix", type=bool, default=False)
    

     # upsampling parameters
    parser.add_argument("--upsample", type=bool, default=False)
    parser.add_argument("--portion_out", type=float, default=0.5)
    parser.add_argument("--upsample_times", type=int, default=1)
    parser.add_argument("--last_feature_path", type=str, default=None)
    parser.add_argument("--last_model_path", type=str, default=None)

    # other setting
    parser.add_argument('--cosine', type=bool, default=False,
                        help='using cosine annealing')
    parser.add_argument('--warm', action='store_true',
                        help='warm-up for large batch training')
    parser.add_argument("--seed", type=int, default=0)

    opt = parser.parse_args()
   
    opt.num_classes = len(osr_splits_inliers[opt.datasets][opt.trail])

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    opt.main_dir = os.getcwd()
    if opt.backbone_model_direct is not None:
        opt.backbone_model_direct = opt.main_dir + opt.backbone_model_direct
        opt.backbone_model_path = os.path.join(opt.backbone_model_direct, opt.backbone_model_name)
        opt.linear_model_path = os.path.join(opt.backbone_model_direct, "_" + opt.datasets + "_last_linear.pth")
    else:
        opt.backbone_model_path = None

    return opt


def load_model(model, path):
    ckpt = torch.load(path, map_location='cpu')
    state_dict = ckpt['model']

    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            model.encoder = torch.nn.DataParallel(model.encoder)
        else:
            new_state_dict = {}
            for k, v in state_dict.items():
                k = k.replace("module.", "")
                new_state_dict[k] = v
            state_dict = new_state_dict
        model = model.cuda()
        cudnn.benchmark = True

        model.load_state_dict(state_dict)
    
    return model


def set_model(opt):
    criterion = torch.nn.CrossEntropyLoss()
    classifier = LinearClassifier(feat_dim=opt.feat_dim, num_classes=opt.num_classes)
    classifier = classifier.cuda()
    criterion = criterion.cuda()

    if opt.datasets == "mnist" or opt.datasets == "FUB":
        in_channels = 1
    else:
        in_channels = 3
        
    if opt.method == "MoCo":
        model = MoCoResNet(opt, name=opt.model, feat_dim=opt.feat_dim, in_channels=in_channels)
    else:
        if opt.model == "resnet18" or opt.model == "resnet34" or opt.model == "resnet50":
            model = SupConResNet(name=opt.model, feat_dim=opt.feat_dim, in_channels=in_channels)
        elif opt.model == "resnet50_pretrain":
            model = ResNet_Pretrain(name='resnet50', feat_dim=opt.feat_dim, freeze_layers=4)
        elif opt.model in ["vgg16", "vgg11", "vgg_s_bn"]:
            model = SupConVGG(name=opt.model, feat_dim=opt.feat_dim, in_channels=in_channels)
        elif opt.model == "MLP":
            model = SupConMLP(feat_dim=opt.feat_dim)
        else:
            model = simCNN_contrastive(opt, feature_dim=opt.feat_dim, in_channels=in_channels)

    if opt.backbone_model_path is not None:
        model = load_model(model, opt.backbone_model_path)
    else:
        model = model.cuda()

    return model, classifier, criterion


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
    for idx, (images, labels) in enumerate(train_loader):
        data_time.update(time.time() - end)

        images = images.cuda(non_blocking=True)
        labels = labels.cuda(non_blocking=True)
        bsz = labels.shape[0]

        # warm-up learning rate
        warmup_learning_rate(opt, epoch, idx, len(train_loader), optimizer)

        # compute loss
        with torch.no_grad():
            if "resnet" in opt.model:
                if opt.method == "MoCo":
                    features = model(images) #model.encoder_q(images)
                else:
                    features = model(images) # model.encoder(images)
            elif "vgg" in opt.model:
                features = model(images)
            elif "simCNN" in opt.model:
                features = model.encoder(images) #model.encoder(images)
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
    classifier.train()

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    with torch.no_grad():
        end = time.time()
        for idx, (images, labels) in enumerate(val_loader):
            images = images.float().cuda()
            labels = labels.cuda()
            bsz = labels.shape[0]

            # forward
            if "resnet" in opt.model:
                if opt.method == "MoCo":
                    features = model(images)
                else:
                    features = model(images)
            elif "vgg" in opt.model:
                features = model(images)
            elif "simCNN" in opt.model:
                features = model.encoder(images)

            output = classifier(features)
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

            if idx % opt.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc1 {top1.val:.3f} ({top1.avg:.3f})'
                      'Acc5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                       idx, len(val_loader), batch_time=batch_time,
                       loss=losses, top1=top1, top5=top5))

    return losses.avg, top1.avg, top5.avg


def main():
    opt = parse_option()

    # build data loader
    train_loader, _ = set_loader(opt)
    
    test_dataset = get_test_datasets(opt)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=opt.batch_size, shuffle=True,
                                              num_workers=opt.num_workers, pin_memory=True)
    # build model and criterion
    model, classifier, criterion = set_model(opt)

    # build optimizer
    optimizer = set_optimizer(opt, classifier)
    
    best_top1_test_acc = 0
    best_top5_test_acc = 0

    # training routine
    for epoch in range(1, opt.epochs + 1):
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

    save_file = opt.backbone_model_name.replace(".pth", "_linear_") + opt.temp_list + ".pth"
    save_file = os.path.join(opt.backbone_model_direct, save_file)
    save_model(model=model, linear=classifier, optimizer=optimizer, opt=opt, epoch=epoch, save_file=save_file)
    print("Best top1 accuacy is ", best_top1_test_acc)
    print("Best top5 accuacy is ", best_top5_test_acc)


if __name__ == '__main__':
    main()
