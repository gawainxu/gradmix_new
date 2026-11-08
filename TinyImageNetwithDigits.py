#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 20:34:36 2024

@author: zhi
"""

import os
import random
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import torch
from torchvision.datasets import MNIST
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torch.nn.functional as F


class TinyImageNetwithDigits(Dataset):
    """
    Defines Tiny Imagenet as for the others pytorch datasets.
    """
    def __init__(self, root="../datasets", classes=range(10), train=True, padding=False, transform=None,
                 target_transform=None, download=True):
        self.not_aug_transform = transforms.Compose([transforms.ToTensor()])
        self.root = root
        self.train = train
        self.padding = padding,
        self.transform = transform
        self.target_transform = target_transform
        self.download = download

        if download:
            if os.path.isdir(root) and len(os.listdir(root)) > 0:
                print('Download not needed, files already on disk.')
            else:
                from google_drive_downloader import GoogleDriveDownloader as gdd

                # https://drive.google.com/file/d/1Sy3ScMBr0F4se8VZ6TAwDYF-nNGAAdxj/view
                print('Downloading dataset')
                gdd.download_file_from_google_drive(
                    file_id='1Sy3ScMBr0F4se8VZ6TAwDYF-nNGAAdxj',

                    dest_path=os.path.join(root, 'tiny-imagenet-processed.zip'),
                    unzip=True)

        self.data = []
        for num in range(20):                                       #20
            self.data.append(np.load(os.path.join(
                root, 'processed/x_%s_%02d.npy' %
                      ('train' if self.train else 'val', num+1))))
        self.data = np.concatenate(np.array(self.data))

        self.targets = []
        for num in range(20):                               # 20
            self.targets.append(np.load(os.path.join(
                root, 'processed/y_%s_%02d.npy' %
                      ('train' if self.train else 'val', num+1))))
        self.targets = np.concatenate(np.array(self.targets))
        
        train_data = []
        train_labels = []

        for i in range(len(self.data)):
            if self.targets[i] in classes:
                train_data.append(self.data[i])
                train_labels.append(self.targets[i])


        self.data = np.array(train_data)
        self.targets = train_labels
        
        self.mnist = mnist()


    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        
        img = np.uint8(255 * img)
        
        if self.train and self.padding:
            mnist_imgs = self.mnist.get_image_class(target.item())
            mnist_img = mnist_imgs[random.randint(0, len(mnist_imgs)-1)]
            img = pad_imgs(img, mnist_img)
        
        img = Image.fromarray(img)                     ## put it in non transform ????????????
        original_img = img.copy()

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)


        if hasattr(self, 'logits'):
            return img, target, original_img, self.logits[index]
        

        return img, target
    
    
def pad_imgs(back, fore):
    
    if len(fore.shape) < 3:
        fore = np.expand_dims(fore, axis=2)   # convert from gray to rgb
        fore = np.repeat(fore, 3, axis=2)
        
    big_size = back.shape[0]
    small_size = fore.shape[0]
    
    anchor_x = random.randint(0, big_size-small_size)
    anchor_y = random.randint(0, big_size-small_size)
    
    fore = np.pad(fore, pad_width=((anchor_x, big_size-small_size-anchor_x), (anchor_y, big_size-small_size-anchor_y), (0, 0)), mode="constant")
    padded_img = np.fmax(fore, back)
    
    return padded_img  
            

    
class mnist(MNIST):
    
    def __init__(self, root="../datasets",
                 classes=range(10),
                 train = True,
                 transform = None,
                 target_transform = None,
                 download = True):
        super(mnist, self).__init__(root, train=train,
                                    transform=transform,
                                    target_transform=target_transform,
                                    download=download)
        
        # Select subset of classes
        if self.train:
            train_data = []
            train_labels = []

            for i in range(0, len(self.data), 10):
                if self.targets[i] in classes:
                    train_data.append(self.data[i])
                    train_labels.append(self.targets[i])

            self.traindata = torch.stack(train_data).numpy()
            self.trainlabels = train_labels

        else:
            test_data = []
            test_labels = []

            for i in range(0, len(self.data), 10):
                if self.targets[i] in classes:
                    test_data.append(self.data[i])
                    test_labels.append(self.targets[i])   # it is torch tensor !!!!!!!!!!!!!

            print(len(test_data))
            self.testdata = torch.stack(test_data).numpy()
            self.testlabels = test_labels
        
        
    def __getitem__(self, index):
        if self.train:
            img, target = self.traindata[index], self.trainlabels[index]
        else:
            img, target = self.testdata[index], self.testlabels[index]

        return img, target
    
    
    def __len__(self):
        if self.train:
            return len(self.traindata)
        else:
            return len(self.testdata)
        
        
    def get_image_class(self, label):
        return self.traindata[np.array(self.trainlabels) == label]
