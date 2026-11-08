#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 20 10:03:47 2025

@author: zhi
"""

import os
import cv2
from voc_base import VOCFull
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import torch
from PIL import ImageFilter, Image
import numpy as np
import scipy
from util import TwoCropTransform

class my_VOC(VOCFull):
    
    """
    return the original voc dataset
    """
    
    def __init__(self, root, train):
        super(my_VOC, self).__init__(root=root, image_set=train, download=True)
                    
    def __getitem__(self, index):
        img, seg_target, dec_target = super().__getitem__(index)
        
        dec_target = dec_target["annotation"]["object"]
        if len(dec_target) > 0:
            dec_target = dec_target[0]
         
        # img is Image and of different sizes    
        return img, dec_target["name"], seg_target


class my_VOC_subsects(Dataset):
    
    """
    return a subset of voc dataset containing the classes in dict
    """
    
    # TODO check the classes
    class_names = ['dog','aeroplane', 'bicycle', "train", 'bird',
                   'boat', 'pottedplant', 'diningtable']
    
    def __init__(self, root, train="train", transform=None, class_names=class_names, mask=True):
        self.myvoc = VOCFull(root)
        self.train = train
        self.ori_len = len(self.myvoc)
        self.class_names = class_names
        self.transform = transform
      
        self.mask = mask
        self.filter_classes()
        
        
    def filter_classes(self):
        
        name_class_dict = {'dog': 0, 'aeroplane': 1, 'bicycle': 2,
                           'train': 3, 'bird': 4, 'boat': 5,
                           'pottedplant': 6, 'diningtable': 7}      
        self.data = []
        self.targets_dec = []
        self.targets_seg = []
        self.seg_bbox = []
        
        # pick the classes in dict
        for i in range(self.ori_len):
            i, dec_name, dec_bbox, l_seg = self.myvoc[i]
            if dec_name in self.class_names:
                self.data.append(i)
                self.targets_dec.append(name_class_dict[dec_name])
                self.targets_seg.append(l_seg)
                self.seg_bbox.append(dec_bbox)

    
    def __getitem__(self, index):
                
        if self.mask:
            print("masking")
            img = my_VOC_subsects.masking(self.data[index], self.targets_seg[index], self.seg_bbox[index])
        else:
            img = self.data[index]
            
        if self.transform is not None:
            return self.transform(img), self.targets_dec[index], self.targets_seg[index]
        else:
            return  img, self.targets_dec[index], self.targets_seg[index]
            
           
    def __len__(self):
        return len(self.data)
    
    @staticmethod
    def masking(img, mask, bbox):
        
        """
        img and mask are both PIL Image
        mask is 0/1 mask
        TODO how if just cut out the object???
        """
        # convert img and mask to np array
        img_np = np.array(img)
        mask_np = np.array(mask)
        # broadcast mask
        mask_np = np.repeat(mask_np[:, :, np.newaxis], 3, axis=2)
        #mask_np_exp = scipy.ndimage.binary_dilation(mask_np)
        # mask * image and filter  --> masked_filtered_img
        processed_mask = np.where(mask_np >= 1, 1, 0)
        img_mask = np.multiply(img_np, processed_mask)
        
        return img_mask


class ImageNet100_masked(Dataset):

    # todo align with the class oders in imagenet-s
    def __init__(self, root, merge=True):

        self.data_train_path = os.path.join(root, "imagenet100_train")
        self.data_test_path = os.path.join(root, "imagenet100_test")
        self.mask_train_path = os.path.join(root, "ImageNetS919/train-semi-segmentation")
        self.mask_test_path = os.path.join(root, "ImageNetS919/validation-segmentation")

        self.train_data = []
        self.train_data_masked = []
        self.train_masks = []
        self.test_data = []
        self.test_data_masked = []
        self.test_masks = []
        self.train_labels = []
        self.test_labels = []

        self.class_dir_list = sorted(os.listdir(self.data_train_path))
        self.mask_dir_list = sorted(os.listdir(self.mask_train_path))
        self.class_map = {serie_num: index for index, serie_num in enumerate(self.class_dir_list)}

        """
        self.transform1 = transforms.Compose([transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
                                             transforms.RandomGrayscale(p=0.2),
                                             transforms.ToTensor(),
                                             transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),])
        self.transform2 = transforms.Compose([transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
                                             transforms.RandomHorizontalFlip(),
                                             transforms.RandomGrayscale(p=0.2),
                                             transforms.ToTensor(),
                                             transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),])
        """
        self.transform1 =  self.transform2 = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),])
        self.transform = TwoCropTransform(transform= self.transform1, transform2=self.transform2)
        self.newsize = 224

        for cd in self.class_dir_list:
            if cd not in self.mask_dir_list:
                continue
            class_data_dir = os.path.join(self.data_train_path, cd)
            class_mask_dir = os.path.join(self.mask_train_path, cd)
            label = self.class_map[cd]
            class_file_list = os.listdir(class_data_dir)
            class_mask_list = os.listdir(class_mask_dir)
            class_mask_list = [n.split(".")[0] for n in class_mask_list]
            for file_name in class_file_list:
                if file_name.split(".")[0] in class_mask_list:
                    im = Image.open(class_data_dir + "/" + file_name)
                    im = im.convert("RGB")
                    mask = Image.open(class_mask_dir + "/" + file_name.split(".")[0] + ".png")
                    mask_np = np.asarray(mask)[:, :, 0]
                    mask_np = (mask_np > 0).astype(int)
                    masked_im = self.masking_img(im, mask_np)
                    self.width, self.height = im.size
                    if self.width <= self.newsize or self.height <= self.newsize:
                        im = im.resize((self.newsize, self.newsize))
                        mask = mask.resize((self.newsize, self.newsize))
                        masked_im = masked_im.resize((self.newsize, self.newsize))
                    else:
                        im = im.crop(((self.width - self.newsize) // 2, (self.height - self.newsize) // 2,
                                      (self.width - self.newsize) // 2 + self.newsize,
                                      (self.height - self.newsize) // 2 + self.newsize))
                        mask = mask.crop(((self.width - self.newsize) // 2, (self.height - self.newsize) // 2,
                                      (self.width - self.newsize) // 2 + self.newsize,
                                      (self.height - self.newsize) // 2 + self.newsize))
                        masked_im = masked_im.crop(((self.width - self.newsize) // 2, (self.height - self.newsize) // 2,
                                                   (self.width - self.newsize) // 2 + self.newsize,
                                                   (self.height - self.newsize) // 2 + self.newsize))
                    mask = np.asarray(mask)[:, :, 0]
                    mask = (mask > 0).astype(int)
                    self.train_data.append(im)
                    self.train_masks.append(mask)
                    self.train_labels.append(label)
                    self.train_data_masked.append(masked_im)

            class_data_dir = os.path.join(self.data_test_path, cd)
            class_mask_dir = os.path.join(self.mask_test_path, cd)
            label = self.class_map[cd]
            class_file_list = os.listdir(class_data_dir)
            class_mask_list = os.listdir(class_mask_dir)
            class_mask_list = [n.split(".")[0] for n in class_mask_list]
            for file_name in class_file_list:
                if file_name.split(".")[0] in class_mask_list:
                    im = Image.open(class_data_dir + "/" + file_name)
                    im = im.convert("RGB")
                    mask = Image.open(class_mask_dir + "/" + file_name.split(".")[0] + ".png")
                    mask_np = np.asarray(mask)[:, :, 0]
                    mask_np = (mask_np > 0).astype(int)
                    masked_im = self.masking_img(im, mask_np)
                    if self.width <= self.newsize or self.height <= self.newsize:
                        im = im.resize((self.newsize, self.newsize))
                        mask = mask.resize((self.newsize, self.newsize))
                        masked_im = masked_im.resize((self.newsize, self.newsize))
                    else:
                        im = im.crop(((self.width - self.newsize) // 2, (self.height - self.newsize) // 2,
                                      (self.width - self.newsize) // 2 + self.newsize,
                                      (self.height - self.newsize) // 2 + self.newsize))
                        mask = mask.crop(((self.width - self.newsize) // 2, (self.height - self.newsize) // 2,
                                      (self.width - self.newsize) // 2 + self.newsize,
                                      (self.height - self.newsize) // 2 + self.newsize))
                        masked_im = masked_im.crop(((self.width - self.newsize) // 2, (self.height - self.newsize) // 2,
                                                    (self.width - self.newsize) // 2 + self.newsize,
                                                    (self.height - self.newsize) // 2 + self.newsize))
                    mask = np.asarray(mask)[:, :, 0]
                    mask = (mask > 0).astype(int)
                    self.test_data.append(im)
                    self.test_masks.append(mask)
                    self.test_labels.append(label)
                    self.test_data_masked.append(masked_im)

        if merge:
            self.train_data = self.train_data + self.test_data
            self.train_masks = self.train_masks + self.test_masks
            self.train_labels = self.train_labels + self.test_labels
            self.train_data_masked = self.train_data_masked + self.test_data_masked

    def __getitem__(self, idx):

        transform_for_ori = transforms.Compose([transforms.ToTensor()])
        transform_for_img = transforms.Compose([transforms.ToTensor(),
                                                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])
        img = self.train_data[idx]
        img_ori = transform_for_ori(img)
        img = self.transform(img)
        img_masked = self.train_data_masked[idx]
        img_masked = self.transform(img_masked)

        return img, img_masked, img_ori, self.train_labels[idx], self.train_masks[idx]

    def __len__(self):

        return len(self.train_data)


    def masking_img(self, img, mask):

        img_np = np.asarray(img)
        mask = np.repeat(mask[:, :, np.newaxis], 3, axis=2).astype(np.uint8)
        img_np = np.multiply(img_np, mask)
        img_np = Image.fromarray(img_np, 'RGB')

        return img_np



if __name__ == "__main__":

    import matplotlib.pyplot as plt
    root = "/home/zhi/projects/datasets"
    my_subsects = ImageNet100_masked(root)
    data_loader = torch.utils.data.DataLoader(my_subsects, batch_size=1, shuffle=False,
                                              num_workers=4, pin_memory=True, sampler=None,
                                              drop_last=True, persistent_workers=True)

    for idx, (images, imgm, labels, masks) in enumerate(data_loader):
        masks = masks.numpy()[0]
        images = images.numpy()[0]
        imgm = imgm.numpy()[0]
        imgm = np.transpose(imgm, axes=[1, 2, 0])
        images = np.transpose(images, axes=[1, 2, 0])
        cv2.imwrite("/home/zhi/projects/temp/"+ str(idx) + "_mask.png",  masks*255)
        cv2.imwrite("/home/zhi/projects/temp/" + str(idx) + "_ori.png",  cv2.cvtColor(images*255, cv2.COLOR_RGB2BGR))
        cv2.imwrite("/home/zhi/projects/temp/" + str(idx) + "_masked.png",  cv2.cvtColor(imgm*255, cv2.COLOR_RGB2BGR))

        