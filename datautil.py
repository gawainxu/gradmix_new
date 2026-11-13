#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 25 21:42:16 2021

@author: zhi
"""

osr_splits_inliers = {
    # Training classes
    'paderborn': [list(range(32)),
                 list(range(16))]}


osr_splits_outliers = {
    # testing classes
    "paderborn": [list(range(32)),
                 list(range(16))]
}



import random
from data_loader import paderborn_machine, BearingContrastiveViews
import torchvision
import torch
from torchvision import transforms
import torch.nn.functional as F
from config import data_root
from losses import SupConLoss
from gradient_cache import gradient_cache_activations

import cv2
import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial.distance import mahalanobis


num_inlier_classes_mapping = {"paderborn": 6}


data_function_mapping = {"paderborn": paderborn_machine, }

data_function_mapping_testing = {"paderborn": paderborn_machine, }

data_function_mapping_outliers = {"paderborn": paderborn_machine, }



mean_mapping = {"paderborn":  (0.1307,),}
std_mapping = {"paderborn":  (0.1307,),}


def label_to_dict(labels, outliers=False):
    """
    Look-up table between original labels and ordered sequential labels
    """
    label_dict = dict()
    for i, l in enumerate(labels):
        if outliers is False:
            label_dict[str(l)] = i
        else:
            label_dict[str(l)] = 1000

    return label_dict


def get_train_datasets(opt):

    data_fun = data_function_mapping[opt.datasets]
    label_dict = label_to_dict(osr_splits_inliers[opt.datasets][opt.trail])

    if opt.action == "training_supcon":
        # TODO settings for train_transform
        train_transform = BearingContrastiveViews()
    else:
        train_transform = None

    classes = osr_splits_inliers[opt.datasets][opt.trail]

    # TODO settings for data_fun
    train = True
    train_dataset = data_fun(root=data_root, train=train, transform=train_transform, target_transform=None,
                             split_mode="window", win_len=4068, step=100, classes=classes,
                             seperate_conditions=False, condition_name="", signal_name="current1")
    print("dataset size", len(train_dataset))
    return train_dataset


def get_test_datasets(opt):

    data_fun = data_function_mapping_testing[opt.datasets]
    classes = osr_splits_inliers[opt.datasets][opt.trail]
    print(classes)
    train = False
    if opt.action == "training_supcon":
        # TODO settings for train_transform and dataset !!!!!!!!!!!!!!
        test_transform = BearingContrastiveViews()
    else:
        test_transform = None
    test_dataset = data_fun(root=data_root, train=train, transform=test_transform, target_transform=None,
                             split_mode="window", win_len=4068, step=100, classes=classes,
                             seperate_conditions=False, condition_name="", signal_name="current1")
    print("dataset size", len(test_dataset))

    return test_dataset



def get_outlier_datasets(opt):

    test_transform = None
    data_fun = data_function_mapping_outliers[opt.datasets]
    label_dict = label_to_dict(osr_splits_outliers[opt.datasets][opt.trail], outliers=True)
    classes = osr_splits_outliers[opt.datasets][opt.trail]
    print(classes)

    train = False
    outlier_dataset = data_fun(root=data_root, train=train, transform=test_transform, target_transform=None,
                             split_mode="window", win_len=4068, step=100, classes=classes,
                             seperate_conditions=False, condition_name="", signal_name="current1")
    print("dataset size", len(outlier_dataset))
    return outlier_dataset



def mahalanobis_group(features, mu, var):

    dis = []
    for feature in features:
        d = mahalanobis(feature, mu, np.linalg.inv(var))
        #print("feature, mu", np.mean(feature), np.mean(mu), np.mean(feature-mu), np.matmul((feature-mu), (feature-mu).T), d)
        dis.append(d)
    
    return dis


def mahalanobis_stats(sorted_last_features, stats, ratio_in, ratio_out):
    
    cut_thresholds = []
    for c, (last_features_c, stats_c) in enumerate(zip(sorted_last_features, stats)):
        mu, var = stats_c
        m_distances = mahalanobis_group(last_features_c, mu, var)        
        m_distances = np.sort(m_distances)
        print("m_distances", np.mean(m_distances))
        cut_index1 = int((1-ratio_out)*len(m_distances))   # outside
        cut_threshold1 = m_distances[cut_index1]
        cut_index2 = int(ratio_in*len(m_distances))         # centers
        cut_threshold2 = m_distances[cut_index2]
        cut_thresholds.append((cut_threshold1, cut_threshold2))

    return cut_thresholds



def mixup_negative(batch_data1, batch_data2, labels1, labels2, alpha, p=0.5):

    lam = np.random.beta(alpha, alpha)     #p = np.random.binomial(1, p)                # np.random.uniform(0.95, 1)
    while lam < 0.8:
        lam = lam = np.random.beta(alpha, alpha)
    new_imgs1 = []
    new_imgs2 = []
    new_labels = []

    # the batch_data is after two_transformation
    batch_data11 = batch_data1
    batch_data12 = batch_data2
    batch_data21 = batch_data1
    batch_data22 = batch_data2
    bz = len(batch_data11)

    for i, (img11, label11) in enumerate(zip(batch_data11, labels1)):
        idx = random.randint(0, bz-1)
        while labels2[idx] == label11:
            idx = random.randint(0, bz-1)
        img12 = batch_data12[idx]
        new_img = lam * img11 + (1 - lam) * img12     # alpha * img11 + (p) * alpha * img12 + (1-p) * alpha * img11    
        new_imgs1.append(new_img)

        img21 = batch_data21[i]
        img22 = batch_data22[idx]
        new_img = lam * img21 + (1 - lam) * img22     # alpha * img21 + (p) * alpha * img22 + (1-p) * alpha * img21     
        new_imgs2.append(new_img)
        new_labels.append([label11.item(), labels2[idx].item()])

    new_imgs1 = torch.stack(new_imgs1)
    new_imgs2 = torch.stack(new_imgs2)
    mask = process_double_label(new_labels)
    
    return [new_imgs1, new_imgs2]


def process_double_label(double_label):

    label_len = len(double_label)
    mask = np.ones([label_len, label_len])

    for i, l1 in enumerate(double_label):
        for j, l2 in enumerate(double_label):
            if len(list(set(l1).intersection(l2))) > 0:
                mask[i, j] = 0

    return mask


def mixup_positive(batch_data1, batch_data2, labels1, labels2, alpha):

    lam = np.random.beta(alpha, alpha)
    new_imgs1 = []
    new_imgs2 = []

    # the batch_data is after two_transformation
    batch_data11 = batch_data1
    batch_data12 = batch_data2
    batch_data21 = batch_data1
    batch_data22 = batch_data2
    bz = len(batch_data11)

    for i, (img11, label11) in enumerate(zip(batch_data11, labels1)):
        idx = random.randrange(0, bz)
        while labels2[idx] != label11:
            idx = random.randrange(0, bz)
        
        img12 = batch_data12[idx]
        new_img = lam * img11 + (1 - lam) * img12
        new_imgs1.append(new_img)

        img21 = batch_data21[i]
        img22 = batch_data22[idx]
        new_img = lam * img21 + (1 - lam) * img22
        new_imgs2.append(new_img)

    new_imgs1 = torch.stack(new_imgs1)
    new_imgs2 = torch.stack(new_imgs2)

    return [batch_data11, batch_data12]

    
def binary_random(p):

     """
     the probability of return 1 is p
     """
     return np.random.binomial(1, p)



def mixup_hybrid_features(batch_features, labels, alpha, beta, batch_features_add = None, labels_add = None, method="min_similarity"):

    """
    mix the positive features with the most dissimilar negative features intra- or 
    inter minibatches

    batch_features1, batch_feature2: tuple of the features in a minibatch of two views
    labels1, labels2: labels of batch_features1, batch_features2 (1 view), bz*1
    alpha, beta: beta distribution coefficients, larger alpha is, smaller beta is, the pdf is more close to 1 
                 with large alpha, then lam is more possible to be close to 1
    """

    batch_features1 = batch_features[0]
    batch_features2 = batch_features[1]
    bz = len(batch_features1)

    batch_features_new1 = []
    batch_features_new2 = []

    if batch_features_add is not None:
        # inter batch mode
        label_mask = torch.eq(labels, labels_add.T).float()
        batch_features_add1 = batch_features_add[0]
        batch_features_add2 = batch_features_add[1]
    else:
        # intra batch mode
        label_mask = torch.eq(labels, labels.T).float()
        batch_features_add1 = batch_features1
        batch_features_add2 = batch_features2

    similarities1 = torch.matmul(batch_features1, batch_features_add1.T)
    similarities1 = similarities1 * label_mask               # mask out these with same labels
                                                             # it's actually not necessary because the same labels won't be the most dissimilar
                                                        
    similarities2 = torch.matmul(batch_features2, batch_features_add2.T)
    similarities2 = similarities2 * label_mask

    for i in range(bz):

        feature1 = batch_features1[i]
        feature2 = batch_features2[i]

        if method == "min_similarity":

            similarity1 = similarities1[i].detach().cpu().numpy()
            masked_zero_similarity1 =  np.ma.masked_where(similarity1 == 0, similarity1)
            idx1 = np.argmin(masked_zero_similarity1)

            similarity2 = similarities2[i].detach().cpu().numpy()
            masked_zero_similarity2 =  np.ma.masked_where(similarity2 == 0, similarity2)
            idx2 = np.argmin(masked_zero_similarity2)

        elif method == "random":

            similarity1 = similarities1[i].detach().cpu().numpy()
            idx1 = np.random.choice(np.where(similarity1>0))

            similarity2 = similarities2[i].detach().cpu().numpy()
            idx2 = np.random.choice(np.where(similarity2>0))

        lam1 = np.random.beta(alpha, beta)
        feature_new1 = lam1 * feature1 + (1-lam1) * batch_features_add1[idx1]
        batch_features_new1.append(feature_new1)

        lam2 = np.random.beta(alpha, beta)
        feature_new2 = lam2 * feature2 + (1-lam2) * batch_features_add2[idx2]
        batch_features_new2.append(feature_new2)

    
    batch_features_new1 = torch.stack(batch_features_new1)
    batch_features_new2 = torch.stack(batch_features_new2)

    return batch_features_new1, batch_features_new2
    

def vanilla_mixup(batch_data1, batch_data2, y, alpha, beta, mode="reverse", encoder=None):

    """
    vanilla mixup within a batch between positives and negatives
    batch_data1, batch_data2 are from two views
    """
    lam = np.random.beta(alpha, beta)
    bsz = batch_data1.shape[0]

    if mode == "random":
        indices = torch.randperm(bsz).cuda()
        indices_pov = indices_neg = indices
    elif mode == "reverse":
        a = [i for i in range(bsz)]
        a.reverse()
        indices = torch.tensor(a).cuda()
        indices_pov = indices_neg = indices
    else:                                                                         #  mode == "max_similarity"
        """
        To mix the samples according to their similarities in feature space
        positive mixup: mix target with the most dissimilar positive sample
        negative mixup: mix target with the most similar negative sample
        The background mechanism is to reduce the amount of feature biased samples, and let the minorities to be emphasised !!!
        How to solve the problem of noisy labels? See how SVHN works without rotation
        """
        features_batch1 = encoder(batch_data1)
        features_batch2 = encoder(batch_data2)

        similarities = torch.matmul(features_batch1, features_batch2.T)
        similarities = similarities.detach().cpu().numpy()
        y = y.detach().cpu().numpy()
        indices = []

        # positive mixup
        for i in range(bsz):
            label = y[i]

            # similarity between features_batch1[i] and all features_batch2
            similarity = similarities[i]

            """
            for mixup in SupCon, select the pairs of different labels and the ones that most 
            similar to the target sample
            """
            ind = np.argsort(similarity)
            while label == y[ind[-1]]:
                # delete the last element which is the index of the current most similar
                ind = ind[:-1] 
            indices.append(ind[-1])

        indices_pov = indices_neg = indices

    batch_data1_new = lam * batch_data1 + (1 - lam) * batch_data2[indices_pov]
    batch_data2_new = lam * batch_data2 + (1 - lam) * batch_data1[indices_neg]

    y_a, y_b = y, y[indices]
    y_new = lam * y_a + (1 - lam) * y_b

    # return lam which will be used for SSL
    return batch_data1_new, batch_data2_new, y_new, lam


def vanilla_cutmix(batch_data1, batch_data2, opt):

    bsz, h, w = batch_data1.shape[0], batch_data1.shape[2], batch_data1.shape[3]
    indices = torch.randperm(bsz).cuda()
    lam = np.random.beta(opt.alpha_vanilla, opt.beta_vanilla)
    while lam > 0.9 or lam < 0.1:
        lam = np.random.beta(opt.alpha_vanilla, opt.beta_vanilla)
    lam = torch.tensor(lam).cuda()

    batch_data1_new = []
    batch_data2_new = []

    for i in range(bsz):
       
        x11 = batch_data1[i]
        x12 = batch_data2[indices[i]]
        x21 = batch_data2[i]
        x22 = batch_data1[indices[i]]
        hl, hh, wl, wh, bbox_area = rand_bbox(img_h=h, img_w=w, lam=lam)
        x11_new = torch.empty_like(x11).copy_(x11)
        x11_new[:, hl:hh, wl:wh] = torchvision.transforms.functional.resize(x12, (hh-hl))
        #hl, hh, wl, wh, bbox_area = rand_bbox(img_h=h, img_w=w, lam=lam)
        x21_new = torch.empty_like(x21).copy_(x21)
        x21_new[:, hl:hh, wl:wh] = torchvision.transforms.functional.resize(x22, (hh-hl))       # , wh-wl

        batch_data1_new.append(x11_new)
        batch_data2_new.append(x21_new)

    new_lam = torch.tensor(1. - bbox_area / h / w).cuda()
    batch_data1_new = torch.stack(batch_data1_new)
    batch_data2_new = torch.stack(batch_data2_new)

    return batch_data1_new, batch_data2_new, new_lam
        
        
def rand_bbox(img_h, img_w, lam, activation=None):
    lam = lam.cpu().numpy()
    ratio = np.sqrt(1 - lam)
    cut_h, cut_w = int(img_h * ratio), int(img_w * ratio)
    if cut_h >= img_h:
        print(img_h, cut_h, ratio)
        
    if activation == None:
        cy = np.random.randint(0 + cut_h // 2, img_h - cut_h // 2)
        cx = np.random.randint(0 + cut_w // 2, img_w - cut_w // 2)
        yl = cy - cut_h // 2
        yh = cy + cut_h // 2
        xl = cx - cut_w // 2
        xh = cx + cut_w // 2
    else:
    
        activation = activation.numpy()
        act_h, act_w = activation.shape[0], activation[1]
        down_ratio = int(img_h / act_h)
        maximum_indices = np.unravel_index(np.argmax(activation, axis=None), activation.shape)
        cy = maximum_indices[0] * down_ratio
        cx = maximum_indices[1] * down_ratio 
        
        yl = np.clip(cy - cut_h // 2, 0, img_h).item()
        yh = np.clip(cy + cut_h // 2, 0, img_h).item()
        xl = np.clip(cx - cut_w // 2, 0, img_w).item()
        xh = np.clip(cx + cut_w // 2, 0, img_w).item()
            

    #print("cy, cx, cut_y, cut_x", cy, cx, cut_h, cut_w)
    return yl, yh, xl, xh, cut_h*cut_w
    

def downsample_data(ori_data, downsample_ratio):

    num_clusters = int(len(ori_data) / downsample_ratio)
    km = KMeans(n_clusters=num_clusters, random_state=0).fit(ori_data)
    downsampled_data = km.cluster_centers_

    return downsampled_data


def salient_maps(batch_data1, batch_data2, model, opt):

    pg = parallel_gradients(model, mode="saliencymix", opt=opt)
    activations, gradients = pg((batch_data1, batch_data2))
    activations, gradients = sort_hooks(activations, gradients)

    pooled_gradients = torch.mean(gradients, dim=[2,3])

    for i in range(activations.size()[1]):
        activations[:, i, :, :] *= pooled_gradients[:, i, None, None]
    
    activations = torch.mean(activations, dim=1)
    activations = F.relu(activations)
    
    return activations


def layer_salient_maps(batch_data1, batch_data2, model, opt):

    bsz, h, w = batch_data1.shape[0], batch_data1.shape[2], batch_data1.shape[3]
    if opt.grad_splits == 1:
       pg = parallel_gradients(model, mode="layersaliencymix", opt=opt)
       activations, gradients = pg((batch_data1, batch_data2))
       activations_list, gradients_list = sort_hooks(activations, gradients)
    else:
       criterion = SupConLoss(temperature=0.05)
       batch_data = torch.cat([batch_data1, batch_data2], dim=0)
       gca = gradient_cache_activations(model=model, splits=opt.grad_splits, loss_fcn=criterion, opt=opt)
       activations, gradients = gca(batch_data)
       activations_list, gradients_list = sort_hooks(activations, gradients)
        
    with torch.no_grad():
        
        if opt.method == "MoCo":
            cam = torch.zeros(1, bsz, h, w)
        else:
            cam = torch.zeros(1, bsz*2, h, w)

        for activations, gradients in zip(activations_list, gradients_list): 
            activation_maps = activations * F.relu(gradients)
            #print("activations", activations.shape, "gradients", gradients.shape)
            cam_layer = torch.sum(activation_maps, dim=1).unsqueeze(0)         # unsqueeze for interpolate function that requires size of [mini-batch x channels x [optional depth] x [optional height] x width]
            cam_layer_min, cam_layer_max = cam_layer.min(), cam_layer.max()
            norm_cam_layer = (cam_layer - cam_layer_min).div(cam_layer_max - cam_layer_min + 1e-8).data
            cam = cam + F.interpolate(norm_cam_layer, size=(h, w), mode="bilinear", align_corners=False)
            
        cam = torch.squeeze(cam)
        
    return cam


def salient_cutmix(batch_data1, batch_data2, model, opt):

    """
    A mixup method that patch on the most activated area of the 
    target image
    1. look for the most activated pixels like in GradCam
    2. Anchor on these pixels to patch the pieces from source images
    """

    bsz, h, w = batch_data1.shape[0], batch_data1.shape[2], batch_data1.shape[3]
    if torch.cuda.is_available() and opt.use_cuda is True:
        indices = torch.randperm(bsz).cuda()
    else:
        indices = torch.randperm(bsz)
    lam = np.random.beta(opt.alpha_vanilla, opt.beta_vanilla)
    lam = lam * (0.9 - 0.5) + 0.5
    #while lam > 0.9 or lam < 0.1:
    #    lam = np.random.beta(opt.alpha_vanilla, opt.beta_vanilla)
    if torch.cuda.is_available() and opt.use_cuda is True:
        lam = torch.tensor(lam).cuda()
    else:
        lam = torch.tensor(lam)
    
    if opt.positive_method == "saliencymix":
        activations = salient_maps(batch_data1, batch_data2, model, opt)
    elif opt.positive_method == "layersaliencymix":    
        activations = layer_salient_maps(batch_data1, batch_data2, model, opt)
    elif opt.positive_method == "cv2saliency":
        activations = conventional_saliency(batch_data1, batch_data2, model, opt)

    batch_data1_new = []
    batch_data2_new = []

    for i in range(bsz):
       
        x11 = batch_data1[i]
        x12 = batch_data2[indices[i]]
        x21 = batch_data2[i]
        x22 = batch_data1[indices[i]]
        activation1 = activations[i]
        #activation2 = activations[i + bsz]           # TODO temperol commented
        hl, hh, wl, wh, bbox_area = rand_bbox(img_h=h, img_w=w, lam=lam, activation=activation1)
        x11_new = torch.empty_like(x11).copy_(x11)
        x21_new = torch.empty_like(x21).copy_(x21)
            
        x11_new[:, hl:hh, wl:wh] = torch.rand(hh-hl, wh-wl)             #torchvision.transforms.functional.resize(x12, (hh-hl, wh-wl))
        #print("hl, hh, wl, wh", hl, hh, wl, wh)
        #hl, hh, wl, wh, bbox_area = rand_bbox(img_h=h, img_w=w, lam=lam, activation=activation2)
        #print("hl, hh, wl, wh", hl, hh, wl, wh)
        #x21[:, hl:hh, wl:wh] = torch.rand(hh-hl, wh-wl)  #torchvision.transforms.functional.resize(x22, (hh-hl, wh-wl))       # , wh-wl

        batch_data1_new.append(x11_new)
        batch_data2_new.append(x21_new)

    if torch.cuda.is_available() and opt.use_cuda is True:
        new_lam = torch.tensor(1. - bbox_area / h / w).cuda()
    else:
        new_lam = torch.tensor(1. - bbox_area / h / w)
    batch_data1_new = torch.stack(batch_data1_new)
    batch_data2_new = torch.stack(batch_data2_new)

    return batch_data1_new, batch_data2_new, new_lam


class LayerTracker:
    def __init__(self):
        # We use OrderedDict to maintain the order in which layers are registered
        self.activations = dict()
        self.gradients = dict()
        self.handles = []

    def _get_hook(self, name):
        # This wrapper creates the actual hook functions
        def forward_hook(module, input, output):
            if name not in self.activations.keys():
                self.activations[name] = dict()
            device_name = output[0].get_device() if isinstance(output, tuple) else str(output.get_device())
            self.activations[name][device_name] = output[0].detach().cpu() if isinstance(output, tuple) else output.detach().cpu()

        def backward_hook(module, grad_input, grad_output):
            if name not in self.gradients.keys():
                self.gradients[name] = dict()
            device_name = grad_output[0].get_device() if isinstance(grad_output, tuple) else str(grad_output.get_device())
            self.gradients[name][device_name] = grad_output[0].detach().cpu() if isinstance(grad_output, tuple) else grad_output.detach().cpu()

        return forward_hook, backward_hook

    def register_layer(self, name, layer):
        f_hook, b_hook = self._get_hook(name)
        h1 = layer.register_forward_hook(f_hook)
        h2 = layer.register_full_backward_hook(b_hook)
        self.handles.extend([h1, h2])

    def remove_hooks(self):
        for handle in self.handles:
            handle.remove()


class parallel_gradients(torch.nn.Module):

    def __init__(self, model, mode="saliencymix", opt=None):
        super(parallel_gradients, self).__init__()
        self.model = model
        self.opt = opt
        self.tracker = LayerTracker()
        
        if torch.cuda.device_count() > 1:
            if opt.method =="MoCo":
                self.encoder = self.model.encoder_q.module
            else:
                self.encoder = self.model.encoder.module
        else:
            if opt.method =="MoCo":
                self.encoder = self.model.encoder_q
            else:
                self.encoder = self.model.encoder

        if self.opt.model == "resnet18":
            self.encoder_layers = [self.encoder.layer1[-1], self.encoder.layer2[-1],
                                   self.encoder.layer3[-1], self.encoder.layer4[-1]]
            self.encoder_layer_names = ["encoder.layer1", "encoder.layer2", "encoder.layer3", "encoder.layer4"]
        elif self.opt.model == "resnet18_pretrain":
            self.encoder_layers = [self.encoder.layer4[-1]]
            self.encoder_layer_names = ["encoder.layer4"]
        elif self.opt.model == "resnet50_pretrain":
            self.encoder_layers = [self.encoder.layer4[-1]]
            self.encoder_layer_names = ["encoder.layer4"]
        elif self.opt.model == "simCNN":
            self.encoder_layers = [self.model.bn10, self.model.bn9,
                                   self.model.bn8, self.model.bn7]
            self.encoder_layer_names = ["bn7", "bn8", "bn9", "bn10"]
        elif self.opt.model == "vgg16" or self.opt.model == "vgg11" or self.opt.model == "vgg_s_bn":
            self.encoder_layers = [self.model.vgg_base.features]
            self.encoder_layer_names = ["vgg_base.features"]

        if opt.method == "MoCo":
            if mode == "saliencymix":
                self.tracker.register_layer(self.encoder_layer_names[-1], self.encoder_layers[-1])
            elif mode == "layersaliencymix":
                self.tracker.register_layer(self.encoder_layer_names[-2], self.encoder_layers[-2])
        else:
            if mode == "saliencymix":
                self.tracker.register_layer(self.encoder_layer_names[-1], self.encoder_layers[-1])
            elif mode == "layersaliencymix":
                for i in self.opt.grad_layers:
                    self.tracker.register_layer(self.encoder_layer_names[i], self.encoder_layers[i])

    def forward(self, x):
        
        if self.opt.method == "SimCLR":
            batch_data1, batch_data2 = x
            bsz = batch_data1.shape[0]
            batch_data = torch.cat([batch_data1, batch_data2], dim=0)
            criterion = SupConLoss(temperature=0.05)
            features = self.model(batch_data)
            features1, features2 = torch.split(features, [bsz, bsz], dim=0)
            features = torch.cat([features1.unsqueeze(1), features2.unsqueeze(1)], dim=1)
            loss = criterion(features)
            loss.backward()
        elif self.opt.method == "MoCo":
            batch_data1, batch_data2 = x
            bsz = batch_data1.shape[0]
            
            feat_q = self.model.head_q(self.model.encoder_q(batch_data1))
            feat_k = self.model.head_k(self.model.encoder_k(batch_data2))
            logits, labels = self.model.InfoNCE_logits(feat_q, feat_k)
            criterion = torch.nn.CrossEntropyLoss()
            
            for param_k in self.model.encoder_k.parameters():
                param_k.requires_grad = False 
            for param_k in self.model.head_k.parameters():
                param_k.requires_grad = False
                
            loss = criterion(logits, labels)
            loss.backward()

        self.tracker.remove_hooks()

        return self.tracker.activations, self.tracker.gradients


def sort_hooks(activations, gradients):

    """
    data of one batch is split sequentially to multiple GPUs
    https://discuss.pytorch.org/t/how-are-batches-split-across-multiple-gpus/91225/2
    """
    activations_np = []
    gradients_np = []

    for k in activations.keys():
        activation_layer = []
        gradient_layer = []
        for device_name in sorted(activations[k].keys()):
            activation = activations[k][device_name]
            activation_layer.append(activation)

        for device_name in sorted(gradients[k].keys()):   # To assure that the device names are aligned for activations and gradients
            gradient = gradients[k][device_name]
            gradient_layer.append(gradient)

        activation_layer = torch.cat(activation_layer, axis=0)
        gradient_layer = torch.cat(gradient_layer, axis=0)
        #print("gradient_layer", gradient_layer.shape)
        #print("activation_layer", activation_layer.shape)

        activations_np.append(activation_layer)
        gradients_np.append(gradient_layer)

    return activations_np, gradients_np


def conventional_saliency(batch_data1, batch_data2, model, opt):
    
    saliency = cv2.saliency.StaticSaliencySpectralResidual_create()
    activations = []
    
    for image in batch_data1:
        image = image.detach().cpu().numpy()
        image = np.transpose(image, (1,2,0))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        (success, saliencyMap) = saliency.computeSaliency(image)
        activations.append(torch.from_numpy(saliencyMap))
        
    return activations
    


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser('argument for training')
    parser.add_argument('--trail', type=int, default=0)
    parser.add_argument('--datasets', type=str, default='paderborn',
                        help='dataset')
    parser.add_argument("--action", type=str, default="training_supcon",
                        choices=["training_supcon", "trainging_linear", "testing_known", "testing_unknown",
                                 "feature_reading"])
    opt = parser.parse_args()

    dataset = get_test_datasets(opt)
    ts, l = dataset[0]