import os
import pickle
import argparse

import torch
import torch.backends.cudnn as cudnn
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset

from datautil import get_train_datasets, get_test_datasets
from networks.resnet_big import SupConResNet, LinearClassifier, ResNet50, MoCoResNet
from networks.maskcon import MaskCon
from networks.simCNN import simCNN_contrastive
from losses import SupConLoss


try:
    import apex
    from apex import amp, optimizers
except ImportError:
    pass


def parse_opts():

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='resnet18')
    parser.add_argument('--datasets', type=str, default='cifar10')
    parser.add_argument('--model_path', type=str,
                        default="./save/SupCon/cifar10_models/cifar10_resnet18_vanilia__SimCLR_1.0_1.0_0.05_trail_0_128_256_old_augmented/last.pth")
    parser.add_argument("--feat_dim", type=int, default=128)
    parser.add_argument('--temp', type=float, default=0.05, help='temperature for loss')

    parser.add_argument('--method', type=str, default='SimCLR',
                        choices=['SupCon', 'SimCLR', "SimCLR_CE", "MoCo"], help='choose method')
    parser.add_argument("--trail", type=int, default=0, choices=[0, 1, 2, 3, 4, 5, 6],
                        help="index of repeating training")
    parser.add_argument("--action", type=str, default="training_supcon",
                        choices=["training_supcon", "trainging_linear", "testing_known", "testing_unknown",
                                 "feature_reading"])
    parser.add_argument('--batch_size', type=int, default=256,
                        help='batch_size')
    parser.add_argument("--downsample_rate", type=int, default=1,)

    parser.add_argument("--use_cuda", type=bool, default=True)
    parser.add_argument('--syncBN', action='store_true',
                        help='using synchronized batch normalization')
    parser.add_argument("--upsample", type=bool, default=False)
    parser.add_argument("--randaug", type=int, default=0)
    parser.add_argument("--augmix", type=bool, default=False)
    parser.add_argument('--num_workers', type=int, default=4,
                        help='num of workers to use')

    opt = parser.parse_args()
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
    train_dataset = Subset(train_dataset, range(0, len(train_dataset), opt.downsample_rate))

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=False,
                                               num_workers=opt.num_workers, pin_memory=True, sampler=train_sampler,
                                               drop_last=True, persistent_workers=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1,
                                              shuffle=False, num_workers=opt.num_workers, pin_memory=True,
                                              sampler=train_sampler, drop_last=True, persistent_workers=True)

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
        if opt.model in ["resnet18", "resnet34", "resnet50"]:
            model = SupConResNet(name=opt.model, feat_dim=opt.feat_dim, in_channels=in_channels)
        elif opt.model == "resnet50_pretrain":
            model = ResNet50(feat_dim=opt.feat_dim, freeze_layers=opt.frozen_layers)
        else:
            model = simCNN_contrastive(opt, feature_dim=opt.feat_dim, in_channels=in_channels)

        criterion1 = SupConLoss(temperature=opt.temp)
        criterion2 = SupConLoss(temperature=opt.temp)
        linear = None

    if opt.model_path is not None:
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

    ckpt = torch.load(opt.model_path, map_location='cpu')
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


def encoder_parameters(encoder):

    return [p for p in encoder.parameters() if p.requires_grad]


def parameters_l2_norm(params):

    return torch.sqrt(sum(torch.sum(p.detach()**2) for p in params))


def average_supervised_direction(model, dataloader, supcon_criterion):

    model.eval()
    params = encoder_parameters(model.encoder)
    num_batches = 0

    gradient_sum = [torch.zeros_like(p) for p in params]

    for idx, (images, labels) in enumerate(dataloader):

        images1 = images[0]
        images2 = images[1]
        images = torch.cat([images1, images2], dim=0)
        bsz = labels.shape[0]
        if torch.cuda.is_available() and opt.use_cuda is True:
            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)

        features = model(images)
        features1, features2 = torch.split(features, [bsz, bsz], dim=0)
        features = torch.cat([features1.unsqueeze(1), features2.unsqueeze(1)], dim=1)
        loss_supcon = supcon_criterion(features, labels)

        gradients = torch.autograd.grad(loss_supcon, params,
                                        retain_graph=False,
                                        create_graph=False,
                                        allow_unused=False)

        for accumulated, gradient in zip(gradient_sum, gradients):
            accumulated.add_(gradient.detach())

        num_batches = idx + 1

    gradient_mean = [gradient / num_batches for gradient in gradient_sum]
    gradient_norm = torch.sqrt(sum(torch.sum(gradient ** 2) for gradient in gradient_mean))
    direction = [gradient / (gradient_norm + 1e-12) for gradient in gradient_mean]

    return direction


@torch.no_grad()
def evaluate_ssl_loss(model, dataloader, ssl_criterion):

    model.eval()

    total_loss = 0
    total_weight = 0

    for idx, (images, labels) in enumerate(dataloader):
        images1 = images[0]
        images2 = images[1]
        images = torch.cat([images1, images2], dim=0)
        bsz = labels.shape[0]
        if torch.cuda.is_available() and opt.use_cuda is True:
            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)

        features = model(images)
        features1, features2 = torch.split(features, [bsz, bsz], dim=0)
        features = torch.cat([features1.unsqueeze(1), features2.unsqueeze(1)], dim=1)
        loss_ssl = ssl_criterion(features)
        total_loss += loss_ssl.cpu().item() * bsz
        total_weight += bsz

    return total_loss / total_weight


@torch.no_grad()
def add_direction(params, direction, step):
    for param, vector in zip(params, direction):
        param.add_(vector, alpha=step)


def directional_ssl_curvature(model, direction, dataloader,
                              ssl_criterion, relative_radius):

    model.eval()
    params = encoder_parameters(model.encoder)

    theta_norm = parameters_l2_norm(params).item()
    epsilon = relative_radius * theta_norm

    loss_zero = evaluate_ssl_loss(model, dataloader, ssl_criterion)
    denominator_loss = abs(loss_zero) + 1e-12

    add_direction(params, direction, epsilon)
    loss_plus = evaluate_ssl_loss(model, dataloader, ssl_criterion)

    add_direction(params, direction, -2.0*epsilon)
    loss_minus = evaluate_ssl_loss(model, dataloader, ssl_criterion)

    # Restore the original parameters.
    add_direction(params, direction, epsilon)
    curvature = (loss_plus + loss_minus - 2 * loss_zero) / (denominator_loss)    # relative_radius ** 2 *

    result = {
        "relative_radius": relative_radius,
        "epsilon": epsilon,
        "loss_zero": loss_zero,
        "loss_plus": loss_plus,
        "loss_minus": loss_minus,
        "curvature": curvature,}

    for key, value in result.items():
        print(f"{key}: {value}")

    return {
        "relative_radius": relative_radius,
        "epsilon": epsilon,
        "loss_zero": loss_zero,
        "loss_plus": loss_plus,
        "loss_minus": loss_minus,
        "curvature": curvature,
    }


if __name__ == "__main__":

    opt = parse_opts()

    model, linear, criterion_supcon, criterion_ssl = set_model(opt)
    train_loader, test_loader = set_loader(opt)

    direction = average_supervised_direction(
        model=model, dataloader=train_loader,
        supcon_criterion = criterion_supcon)

    radii = [1e-5, 3e-5, 1e-4, 3e-4, 1e-3]

    results = [
        directional_ssl_curvature(
            model=model,
            direction=direction,
            dataloader=train_loader,
            ssl_criterion=criterion_ssl,
            relative_radius=radius)
        for radius in radii
    ]

    curvature = 0
    for result in results:
        for key, value in result.items():
            print(f"{key}: {value}")
        curvature += result["curvature"]

    print("curvature mean", curvature/len(radii))

