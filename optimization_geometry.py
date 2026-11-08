import os
import pickle
import argparse

import torch
import torch.backends.cudnn as cudnn
from torchvision import transforms
import torch.nn.functional as F
from torch.autograd.functional import hessian
from torch.utils.data import Subset

import numpy as np
try:
    import apex
    from apex import amp, optimizers
except ImportError:
    pass

from networks.resnet_big import SupConResNet
from losses import SupConLoss
from datautil import mean_mapping, std_mapping, image_size_mapping, data_function_mapping
from datautil import label_to_dict, osr_splits_inliers
from util import TwoCropTransform
from datautil import get_train_datasets


def parse_options():
    parser = argparse.ArgumentParser('argument for evaluation')

    parser.add_argument("--models_path", type=str, default="/cifar100_models/cifar100_resnet18_vanilia__SimCLR_1.0_1.0_0.05_trail_0_128_256/")
    parser.add_argument('--model', type=str, default='resnet18',
                        choices=["resnet18", "resnet34", "resnet50", "preactresnet18", "preactresnet34", "simCNN",
                                 "MLP"])
    parser.add_argument('--temp', type=float, default=0.05, help='temperature for loss')
    parser.add_argument("--feat_dim", type=int, default=128)
    parser.add_argument("--datasets", type=str, default="imagenet100_m")
    parser.add_argument("--data_root", type=str, default="../datasets")
    parser.add_argument("--trail", type=int, default=0, choices=[0, 1, 2, 3, 4, 5, 6],
                        help="index of repeating training")
    parser.add_argument("--save_path", type=str, default="./")
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--epoch_step", type=int, default=10)
    parser.add_argument("--train_or_test", type=str, default="train",
                        help="which data to use, train or test")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--use_cuda", type=bool, default=False)
    parser.add_argument('--syncBN', action='store_true',
                        help='using synchronized batch normalization')
    parser.add_argument("--action", type=str, default="training_supcon",
                       choices=["training_supcon", "trainging_linear", "testing_known", "testing_unknown", "feature_reading"])
    parser.add_argument("--augmix", type=bool, default=False) 
    parser.add_argument("--randaug", type=int, default=0)

    opt = parser.parse_args()

    if torch.cuda.is_available() and opt.use_cuda == "gpu":
        opt.device = "cuda"
    else:
        opt.device = "cpu"

    return opt


def set_model(opt):

    in_channels = 3
    model = SupConResNet(name=opt.model, feat_dim=opt.feat_dim, in_channels=in_channels)

    criterion1 = SupConLoss(temperature=opt.temp)
    criterion2 = SupConLoss(temperature=opt.temp)

    # enable synchronized Batch Normalization
    if opt.syncBN:
        model = apex.parallel.convert_syncbn_model(model)

    if opt.device == "cuda":
        if torch.cuda.device_count() > 1:
            model.encoder = torch.nn.DataParallel(model.encoder)
        model = model.cuda()

        if criterion1 is not None:
            criterion1 = criterion1.cuda()
        if criterion2 is not None:
            criterion2 = criterion2.cuda()
        cudnn.benchmark = True

    return model, criterion1, criterion2


def load_model(opt, model):

    ckpt = torch.load(opt.model_path, map_location='cpu')
    state_dict = ckpt['model']

    new_state_dict = {}
    for k, v in state_dict.items():
        k = k.replace("module.", "")
        new_state_dict[k] = v

    state_dict = new_state_dict
    model.load_state_dict(state_dict)
    model.to(opt.device)
    model.eval()

    return model


def set_loader(opt):

    dataset = get_train_datasets(opt)

    if opt.datasets != "imagenet100":
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size, shuffle=False,
                                                   num_workers=1, pin_memory=True, sampler=None,
                                                   drop_last=True,
                                                   persistent_workers=True)
    else:
        data_loader = dataset

    return data_loader


def get_datasets(opt):
    mean = mean_mapping[opt.datasets]
    std = std_mapping[opt.datasets]
    normalize = transforms.Normalize(mean=mean, std=std)
    size = image_size_mapping[opt.datasets]

    transform = transforms.Compose(
                [transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
                 transforms.Resize((size, size)),
                 transforms.RandomHorizontalFlip(),
                 transforms.RandomGrayscale(p=0.2),
                 transforms.ToTensor(),
                 normalize, ])

    data_fun = data_function_mapping[opt.datasets]
    label_dict = label_to_dict(osr_splits_inliers[opt.datasets][opt.trail])
    transform = TwoCropTransform(transform)
    classes = osr_splits_inliers[opt.datasets][opt.trail]

    if opt.train_or_test == "train":
        train = True
    else:
        train = False

    dataset = data_fun(root=opt.data_root, train=train,
                       classes=classes, download=True,
                       transform=transform, label_dict=label_dict)
    print("dataset size", len(dataset))
    return dataset


def hessian_single_layer(model, x, labels, bsz, criterion1, criterion2):

    last_linear = model.head[-1]
    weight_matrix = last_linear.weight.clone()
    bias = last_linear.bias.clone()

    def compute_inpt(x, model):
        model.eval()
        inpt = model.encoder(x)
        inpt = model.head[0](inpt)
        inpt = model.head[1](inpt)

        return inpt

    def custome_forward(x):
        features = F.linear(inpt, weight_matrix, bias)
        features = F.normalize(features, dim=1)
        features1, features2 = torch.split(features, [bsz, bsz], dim=0)
        features = torch.cat([features1.unsqueeze(1), features2.unsqueeze(1)], dim=1)
        loss_sup = criterion2(features, labels)
        loss_ssl = criterion1(features)

        return loss_sup

    inpt = compute_inpt(x, model)
    h_sup = hessian(custome_forward, inpt)
    loss_sup = custome_forward(inpt)
    g_sup = torch.autograd.grad(loss_sup, weight_matrix,
                                retain_graph=True)[0]
    for g in g_sup:
        h =  torch.autograd.grad(g, weight_matrix, allow_unused=True,
                                 retain_graph=True)[0]

    return h_sup



def grad_projection(grad1, grad2):

    # to project g1 to g2
    prod = np.inner(grad1, grad2)
    prod = prod / np.linalg.norm(grad2, axis=1) / np.linalg.norm(grad2, axis=1)
    prod = np.average(prod)
    #proj = prod * grad2

    return prod



def main(opt):

    data_loader = set_loader(opt)
    model, criterion1, criterion2 = set_model(opt)

    losses_sup_all = []
    losses_ssl_all = []
    g_sup_all = []
    g_ssl_all = []
    h_sup_all = []
    h_ssl_all = []

    for epoch in range(10, opt.num_epochs, opt.epoch_step):

        print("epoch", epoch)
        opt.model_path = os.path.join(opt.models_path, "ckpt_epoch_{}.pth".format(epoch))
        model = load_model(opt, model)

        losses_sup_epoch = []
        losses_ssl_epoch = []
        g_sup_epoch = []
        g_ssl_epoch = []
        h_sup_epoch = []
        h_ssl_epoch = []
        angles = []
        
        for idx, (images, labels) in enumerate(data_loader):

            if idx > 20:
                break

            images1 = images[0]
            images2 = images[1]
            bsz = labels.shape[0]

            if opt.device == "cuda":
                images1 = images1.cuda()
                images2 = images2.cuda()

            images = torch.cat([images1, images2], dim=0)
            features = model(images)
            features1, features2 = torch.split(features, [bsz, bsz], dim=0)
            features = torch.cat([features1.unsqueeze(1), features2.unsqueeze(1)], dim=1)

            loss_sup = criterion2(features, labels)
            loss_ssl = criterion1(features)

            print(loss_sup, loss_ssl)

            g_sup = torch.autograd.grad(loss_sup, model.head[2].weight,
                                        retain_graph=True)[0]
            g_ssl = torch.autograd.grad(loss_ssl, model.head[2].weight,
                                        retain_graph=True)[0]

            #g_sup.requires_grad_(True)
            #g_ssl.requires_grad_(True)
            #h_sup, h_ssl = hessian_single_layer(model, images, labels, bsz, criterion1, criterion2)
            """
            flattened_grads = torch.cat(([grad.flatten() for grad in g_sup]))
            hessian = torch.zeros(flattened_grads.shape[0], flattened_grads.shape[0])
            for idx, grad in enumerate(g_sup):
                second_der = torch.autograd.grad(grad, model.head[2].weight, retain_graph=True, allow_unused=True)
                second_der = torch.cat(([grad.flatten() for grad in second_der]))
                hessian[idx, :] = second_der
            """

            losses_sup_epoch.append(loss_sup.detach().cpu().numpy())
            losses_ssl_epoch.append(loss_ssl.detach().cpu().numpy())
            g_sup = g_sup.detach().cpu().numpy()
            g_ssl = g_ssl.detach().cpu().numpy()
            angle = grad_projection(g_sup, g_ssl)
            g_sup_epoch.append(g_sup)
            g_ssl_epoch.append(g_ssl)
            angles.append(angle)
            print("g_sup", np.sum(np.abs(g_sup)))
            print("g_ssl", np.sum(np.abs(g_ssl)))
            print("angle", angle)
            #h_sup_epoch.append(h_sup.detach().cpu().numpy())
            #h_ssl_epoch.append(h_ssl.detach().cpu().numpy())

        opt.save_path = os.path.join(opt.models_path, "gradients_{}".format(epoch))
        with open(opt.save_path, "wb") as f:
            pickle.dump((losses_sup_epoch, losses_ssl_epoch, g_sup_epoch, g_ssl_epoch), f)

        losses_sup_all.append(losses_sup_epoch)
        losses_ssl_all.append(losses_ssl_epoch)
        g_sup_all.append(g_sup_epoch)
        g_ssl_all.append(g_ssl_epoch)
        print("Average Angle", sum(angles)/len(angles))
        #h_sup_all.append(h_sup_epoch)
        #h_ssl_all.append(h_ssl_epoch)





if __name__ == "__main__":

    opt = parse_options()
    main(opt)

    # following steps need to be done
    # masked dataset
    # gradient projection


