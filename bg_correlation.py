#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 11 18:00:26 2025

@author: zhi
"""

import os
import Path
import argparse
import pickle
import numpy as np
from scipy.stats import entropy
from scipy.ndimage import binary_dilation, generate_binary_structure

import torch
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
from PIL import Image

from masked_datasets import ImageNet100_masked
from networks.resnet_big import SupConResNet, LinearClassifier

# global variables
global gradients
gradients = {}

global activations
activations = {}


def parse_option():
    parser = argparse.ArgumentParser('argument for BG analysis')

    # model dataset    
    parser.add_argument('--model', type=str, default='resnet18', choices=["resnet18", "resnet34"])
    parser.add_argument("--feat_dim", type=int, default=128)
    parser.add_argument("--bsz", type=int, default=1)
    parser.add_argument("--data_root", type=str, default="datasets")
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--num_classes", type=int, default=100)
    parser.add_argument("--backbone_model_direct", type=str,
                        default="/save/SupCon/imagenet100_m_models/imagenet100_m_resnet18_vanilia__SimCLR_1.0_1.0_0.05_trail_0_128_256")
    parser.add_argument("--backbone_model_name", type=str, default="last.pth")
    parser.add_argument("--linear_model_name", type=str, default="last_linear.pth")
    parser.add_argument("--output_path", type=str, default="./features/energy_entropy")
    
    parser.add_argument("--threshold", type=float, default=0.01)
    parser.add_argument("--method", type=str, default="ssl")
    parser.add_argument("--if_train", type=bool, default=True)

    opt = parser.parse_args()

    opt.backbone_model_path = os.path.join(opt.backbone_model_direct, opt.backbone_model_name)
    opt.linear_model_path = os.path.join(opt.backbone_model_direct, opt.linear_model_name)

    opt.main_dir = os.getcwd()
    opt.parent_dir = Path(opt.main_dir).parent.absolute()
    opt.data_root = os.path.join(opt.parent_dir, opt.data_root)
    
    opt.output_path = opt.output_path + "_" + opt.method + "_" + str(opt.threshold) + "_" + str(opt.if_train)

    return opt



def load_model(model, classifier, opt):
    ckpt = torch.load(opt.backbone_model_path, map_location='cpu')
    state_dict = ckpt['model']

    linear_ckpt = torch.load(opt.linear_model_path, map_location="cpu")
    state_dict_classifier = linear_ckpt["linear"]

    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            model.encoder = torch.nn.DataParallel(model.encoder)
        else:
            new_state_dict = {}
            new_state_dict_classifier = {}
            for k, v in state_dict.items():
                k = k.replace("module.", "")
                new_state_dict[k] = v
            for k, v in state_dict_classifier.items():
                k = k.replace("module.", "")
                new_state_dict_classifier[k] = v

            state_dict = new_state_dict
            state_dict_classifier = new_state_dict_classifier
        
        model = model.cuda()
        classifier = classifier.cuda()
        cudnn.benchmark = True

        model.load_state_dict(state_dict)
        classifier.load_state_dict(state_dict_classifier)
    
    return model, classifier


def set_model(opt):
    criterion = torch.nn.CrossEntropyLoss()
    classifier = LinearClassifier(name=opt.model, num_classes=opt.num_classes)
    classifier = classifier.cuda()
    criterion = criterion.cuda()

    model = SupConResNet(name=opt.model, feat_dim=opt.feat_dim)
    model, classifier = load_model(model, classifier, opt)
    
    model.eval()
    classifier.eval()

    return model, classifier


def compute_joint_rgb_entropy(flatted_img, r=16):

    flatted_img = np.array(flatted_img)
    print("flatted_img", flatted_img.shape)
    quantized = flatted_img * (256 // r)
    quantized = quantized.astype(int)
    print("quantized", quantized.shape)
    
    # Convert each RGB triplet to a single index
    indices = quantized[:, 0] * (256 // r)**2 + quantized[:, 1] * (256 // r) + quantized[:, 2]
    
    hist, _ = np.histogram(indices, bins=(256 // r)**3, range=(0, (256 // r)**3), density=True)
    hist = hist[hist > 0]
    
    return entropy(hist, base=2)


def compute_pixel_entropy(img, mask, idx=0):
  
    #transform = transforms.Grayscale()
    #img = transform(img)
    flatted_img = flatten_image_with_mask(img, mask, idx)
    if len(flatted_img) > 0:
        enp = compute_joint_rgb_entropy(flatted_img)
        return enp
    else:
        return 1000
    
    #hist, _ = np.histogram(flatted_img, bins=256, range=(0, 1), density=True)
    #hist = hist[hist > 0]  # remove zeros to avoid log(0)


def flatten_image_with_mask(img, mask, idx=0):
    
    img = img.cpu()
    img, mask = np.array(img), np.array(mask)
    
    """
    m = np.repeat(mask[np.newaxis,:,:], 3, axis=0)
    im = np.squeeze(img)*m
    im = im * 256
    im=np.transpose(im, axes=[1,2,0]).astype(np.uint8)
    im = Image.fromarray(im)
    im.save(str(idx) + ".jpeg")
    """
    
    mask = np.squeeze(mask)
    img = np.squeeze(img)
    
    # in case for gray images
    if img.shape[0] == 1:
        img = np.concatenate([img, img, img], axis=0)
        
    flatten_img = []
    img = img.reshape([3, -1])
    print("mask", np.sum(mask))
    
    for i, p_mask in enumerate(mask.flatten()):
        if p_mask == 0:
            flatten_img.append(img[:, i])
    
    return flatten_img


def threshold_energy(cam, mask, threshold=0):
    
    cam_norm = (cam-cam.min()) / (cam.max() - cam.min())
    cam_norm = np.where(cam_norm>threshold, 1, 0)
    struct = generate_binary_structure(2, 2)
    mask_np_exp = binary_dilation(mask, structure=struct, iterations=10)
    
    area_obj = np.sum(mask_np_exp)
    area_in = np.sum(cam_norm * mask_np_exp)
    area_out = np.sum((1-mask_np_exp)*cam_norm)
    
    energy = area_in / area_obj #area_out #(area_obj + area_out)
    
    return energy


def backward_hook(module, grad_input, grad_output):
    
    #print("Backward Hook Running", grad_output[0].shape)
    gradients[str(4-grad_output[0].shape[-1]/8)] = grad_output
    
    
def forward_hook(module, input, output):
    
    #print("Forward Hook Running", output[0].shape)
    activations[str(4-output[0].shape[-1]/8)] = output



if __name__ == "__main__":
    
    opt = parse_option()
    
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225))])
    datasets = ImageNet100_masked(root=opt.data_root)
    print("len dataset", len(datasets))
    data_loader = DataLoader(datasets, batch_size=opt.bsz, shuffle=False, num_workers=1, pin_memory=True)
    
    model, classifier = set_model(opt)
    criterion = torch.nn.CrossEntropyLoss()
    
    # register hook
    hooks = []
    hooks.append(model.encoder.layer4[-1].register_forward_hook(hook=forward_hook) )
    hooks.append(model.encoder.layer4[-1].register_full_backward_hook(hook=backward_hook)) 
    
    hooks.append(model.encoder.layer3[-1].register_forward_hook(hook=forward_hook) )
    hooks.append(model.encoder.layer3[-1].register_full_backward_hook(hook=backward_hook)) 
    
    hooks.append(model.encoder.layer2[-1].register_forward_hook(hook=forward_hook) )
    hooks.append(model.encoder.layer2[-1].register_full_backward_hook(hook=backward_hook)) 
    
    energy_list = []
    bg_entropy_list = []
    
    for idx, (img, img_masked, img_ori, label, mask) in enumerate(data_loader):
        
        img = img.cuda()
        label = label.cuda()
        features = model.encoder(img)
        output = classifier(features)
        loss = criterion(output, label)
        
        #print("loss", loss)
        loss.backward()
        
        cam = torch.zeros(opt.bsz, 1, opt.img_size, opt.img_size)
        for key in activations.keys():
            
            print("keys", key)
            if key != "0.5":
                continue
            
            activations_layer = activations[key]
            gradients_layer = gradients[key][0]
            
            #print(activations_layer)
            #print(gradients_layer)
            
            for i in range(opt.bsz):
                #print("i", i)
                activation_i = activations_layer[i]
                gradient_i = gradients_layer[i]
                with torch.no_grad():
                    activation_maps = activation_i * F.relu(gradient_i)
                    cam_layer = torch.sum(activation_maps, dim=0).unsqueeze(0).unsqueeze(0)
                    cam_layer = cam_layer.cpu()
                    #print("cam_layer", torch.sum(cam_layer))
                    cam[i] = cam[i] + F.interpolate(cam_layer, size=(opt.img_size, opt.img_size), mode="bilinear", align_corners=False)[0]
        
        cam = np.squeeze(np.array(cam))
        mask = np.squeeze(np.asarray(mask))[0,:,:]
        
        #energy = np.sum(cam * mask ) / np.sum(cam)
        energy = threshold_energy(cam, mask, threshold=opt.threshold)
        print("energy is", energy)
        energy_list.append(energy)
        
        #bg_entropy = compute_pixel_entropy(img_ori, mask, idx=idx)
        #bg_entropy_list.append(bg_entropy)
    
    #with open(opt.output_path, "wb") as f:
    #    pickle.dump((bg_entropy_list, energy_list), f)
        
        
        
        
        
    
    
    