import matplotlib.pyplot as plt
from torchvision.datasets import CIFAR10, CIFAR100, MNIST, SVHN
from torchvision.datasets import VisionDataset
from torchvision.datasets.folder import default_loader
from torch.utils.data.dataloader import default_collate
from datasets import load_dataset
from torchvision import transforms

import datautil
import numpy as np
import os
import gzip
import torch
import struct
import pickle
import PIL
import csv

import torchvision.transforms as transforms
from util import common_elements, TwoCropTransform
from torch.utils.data import Dataset
import torch.utils.data as data
from torchvision.datasets import ImageFolder
import xml.etree.ElementTree as ET


class iCIFAR10(CIFAR10):
    
    def __init__(self, root, classes=range(10), train=True, opt=None, transform=None,
                 target_transform=None, download=False, label_dict = None, last_features_list=None, 
                 last_feature_labels_list=None, last_model=None, subsample_transform=None, portion_out=0.1, upsample_times=1):
        super(iCIFAR10, self).__init__(root,
                                       train=train,
                                       transform=transform,
                                       target_transform=target_transform,
                                       download=download)
        self.label_dict = label_dict

        # Select subset of classes
        if self.train:
            train_data = []
            train_labels = []

            for c in classes:
                for i in range(len(self.data)):
                    if self.targets[i] == c:
                        train_data.append(self.data[i])
                        train_labels.append(self.targets[i])

            final_indices = []
            if subsample_transform is not None:

                #train_data, train_labels = datautil.sample_distance_center_mahalanobis(train_data=train_data, train_labels=train_labels, last_features=last_features, last_feature_labels=last_feature_labels, last_model=last_model,
                #                                                                       label_dict=label_dict, num_classes=len(classes), subsample_transform=subsample_transform, ratio_in=ratio_center, ratio_out=ratio, dataset_name="cifar10")
                for idx, (last_features, last_feature_labels) in enumerate(zip(last_features_list, last_feature_labels_list)):
                    kept_indices = datautil.upsample_distance_center_mahalanobis(train_data=train_data, train_labels=train_labels, last_features=last_features, last_feature_labels=last_feature_labels, last_model=last_model,
                                                                                 label_dict=label_dict, num_classes=len(classes), subsample_transform=subsample_transform, portion_out=portion_out, dataset_name="cifar10", 
                                                                                 upsample_times=upsample_times, idx=idx)
                    if idx == 0:
                        final_indices = kept_indices
                    else:
                        final_indices = common_elements(final_indices, kept_indices)
                    
                train_data = [train_data[i] for i in final_indices] 
                train_labels = [train_labels[i] for i in final_indices] 

            self.train_data = np.array(train_data)
            self.train_labels = train_labels

            print("Final Data Size ", len(self.train_data))

        else:
            test_data = []
            test_labels = []

            """
            for i in range(len(self.data)):
                if self.targets[i] in classes:
                    test_data.append(self.data[i])
                    test_labels.append(self.targets[i])
            """

            for c in classes:
                for i in range(len(self.data)):
                    if self.targets[i] == c:
                        test_data.append(self.data[i])
                        test_labels.append(self.targets[i])

            self.test_data = np.array(test_data)
            self.test_labels = test_labels

    def __getitem__(self, index):
        if self.train:
            img, target = self.train_data[index], self.train_labels[index]
        else:
            img, target = self.test_data[index], self.test_labels[index]

        if self.transform is not None:
            img = PIL.Image.fromarray(img)
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)
        
        if self.label_dict is not None:
            target = self.label_dict[str(target)]

        return img, target

    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)

    def get_image_class(self, label):
        return self.train_data[np.array(self.train_labels) == label]
    
    
    def get_part_data(self, xidxs):
        
        self.train_data = np.delete(self.train_data, xidxs, 0)
        self.train_labels = np.delete(self.train_labels, xidxs, 0)


    def append(self, images, labels):
        """Append dataset with images and labels

        Args:
            images: Tensor of shape (N, C, H, W)
            labels: list of labels
        """

        self.train_data = np.concatenate((self.train_data, images), axis=0)
        self.train_labels = self.train_labels + labels


class iCIFAR100(CIFAR100):
    def __init__(self, root,
                 classes=range(100),
                 opt=None,
                 superClass = None,
                 train=True,
                 transform=None,
                 target_transform=None,
                 download=False,
                 label_dict = None, last_features_list=None, 
                 last_feature_labels_list=None, last_model=None, subsample_transform=None, portion_out=0.1, upsample_times=1):
        super(iCIFAR100, self).__init__(root,
                                        train=train,
                                        transform=transform,
                                        target_transform=target_transform,
                                        download=download)
        self.label_dict = label_dict

        if superClass is not None:
            classes = [datautil.classMap[n] for n in datautil.superClasses[superClass]]

        # Select subset of classes
        if self.train:
            train_data = []
            train_labels = []

            for i in range(len(self.data)):
                if self.targets[i] in classes:
                    train_data.append(self.data[i])
                    train_labels.append(self.targets[i])

            self.train_data = np.array(train_data)
            self.train_labels = train_labels
            
        else:
            test_data = []
            test_labels = []

            for i in range(len(self.data)):
                if self.targets[i] in classes:
                    test_data.append(self.data[i])
                    test_labels.append(self.targets[i])

            self.test_data = np.array(test_data)
            self.test_labels = test_labels

    def __getitem__(self, index):
        if self.train:
            img, target = self.train_data[index], self.train_labels[index]
        else:
            img, target = self.test_data[index], self.test_labels[index]

        if self.transform is not None:
            img = PIL.Image.fromarray(img)
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        if self.label_dict is not None:
            target = self.label_dict[str(target)]

        return img, target

    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)

    def get_image_class(self, label):
        return self.train_data[np.array(self.train_labels) == label]
    
    
    def get_part_data(self, xidxs):
        
        self.train_data = np.delete(self.train_data, xidxs, 0)
        self.train_labels = np.delete(self.train_labels, xidxs, 0)


    def append(self, images, labels):
        """Append dataset with images and labels

        Args:
            images: Tensor of shape (N, C, H, W)
            labels: list of labels
        """

        self.train_data = np.concatenate((self.train_data, images), axis=0)
        self.train_labels = self.train_labels + labels        


class mnist(MNIST):
    
    def __init__(self, root,
                 classes=range(10),
                 train = True,
                 opt=None,
                 transform = None,
                 target_transform = None,
                 download = True,
                 label_dict = None, last_features_list=None, 
                 last_feature_labels_list=None, last_model=None, 
                 subsample_transform=None, portion_out=0.1, upsample_times=1):
        super(mnist, self).__init__(root, train=train,
                                    transform=transform,
                                    target_transform=target_transform,
                                    download=download)
        
        self.label_dict = label_dict
        # Select subset of classes
        if self.train:
            train_data = []
            train_labels = []

            for i in range(0, len(self.data)):    # subsample
                if self.targets[i] in classes:
                    train_data.append(self.data[i])
                    train_labels.append(self.targets[i])

            self.traindata = torch.stack(train_data).numpy()
            self.trainlabels = train_labels

        else:
            test_data = []
            test_labels = []

            for i in range(0, len(self.data)):
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

        img = PIL.Image.fromarray(img, mode='L')
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        if self.label_dict is not None:
            target = self.label_dict[str(target.item())]

        return img, target
    
    
    def __len__(self):
        if self.train:
            return len(self.traindata)
        else:
            return len(self.testdata)
        
        
    def get_image_class(self, label):
        return self.traindata[np.array(self.trainlabels) == label]
    
    
class mnist1(data.Dataset):


      datas = ['train-images-idx3-ubyte.gz',
               'train-labels-idx1-ubyte.gz',
               't10k-images-idx3-ubyte.gz',
               't10k-labels-idx1-ubyte.gz']

      taining_file = 'training.pt'
      test_file = 'test.pt'

      def __init__(self, root, classes=range(10), opt=None, train=True, download=False,
                   transform=None, target_transform=None):
   
          self.root = os.path.expanduser(root)
          self.train = train
          self.transform = transform
          self.target_transform = target_transform

          if self.train:
             self.train_data = self.read_img_file(root + '/mnist/raw/train-images-idx3-ubyte.gz')
             self.train_labels = self.read_label_file(root + '/mnist/raw/train-labels-idx1-ubyte.gz')
             
             train_data = []
             train_labels = []

             for i in range(0, len(self.train_data)):
                 if self.train_labels[i] in classes:
                     train_data.append(self.train_data[i])
                     train_labels.append(self.train_labels[i])

             self.train_data = torch.stack(train_data).numpy()
             self.train_labels = train_labels
             
          else:
             self.test_data =  self.read_img_file(root + '/mnist/raw/train-images-idx3-ubyte.gz')
             self.test_labels =  self.read_label_file(root + '/mnist/raw/train-labels-idx1-ubyte.gz')             
             
             test_data = []
             test_labels = []

             for i in range(len(self.test_data), 10):
                 if self.test_labels[i] in classes:
                     test_data.append(self.test_data[i])
                     test_labels.append(self.test_labels[i])   # it is torch tensor !!!!!!!!!!!!!

             print(len(test_data))
             self.test_data = torch.stack(test_data).numpy()
             self.test_labels = test_labels

       
      def __getitem__(self, index):
          """
          Args:
              index(int): Index
          Returns:
              tuple: (image, target) where target is index of the target class
          """
          if self.train:
             img, target = self.train_data[index], self.train_labels[index]
          else:
             img, target = self.test_data[index], self.test_labels[index]

          if self.transform is not None:
             img = PIL.Image.fromarray(np.squeeze(img), mode='L')
             img = self.transform(img)

          if self.target_transform is not None:
             target = self.target_transform(target)

          return img, target


      def __len__(self):
          if self.train:
             return len(self.train_data)
          else:
             return len(self.test_data) 
         
      def get_image_class(self, label):
          if self.train:
              return self.train_data[np.array(self.train_labels) == label]
          else:
              return self.test_data[np.array(self.test_labels) == label]
         
            
      def read_label_file(self, path):
          # read all images at once  
          # return: torch tensor  
          
          f = gzip.open(path, 'rb')
          f.read(8)                  # 4 byte integer magic number, 4 byte number of items
          if self.train:
              num_imgs = 10000        #struct.unpack('>I', f.read(4))[0]  !!!!!!!!!!!
          else:
              num_imgs = 10000
    
          buf = f.read(num_imgs)
          label = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
          label_torch = torch.from_numpy(label).view(num_imgs).long()

          return label_torch
      
        
      def read_img_file(self, path):
          # read all images at once
          # return: torch tensor
          f = gzip.open(path, 'rb')
          f.read(8)                                         # skip the first 16 bytes, 4 byte integer magic number, 
          if self.train:
              num_imgs = 10000                                  #struct.unpack('>I', f.read(4))[0]       # 4 byte integer number of images, !!!!!!!!!!!!!!! 
          else:
              num_imgs = 10000                  
          num_rows = struct.unpack('>I', f.read(4))[0]      # 4 byte number of rows
          num_cols = struct.unpack('>I', f.read(4))[0]      # 4 byte number of columns
          buf = f.read(num_cols*num_rows*num_imgs)
          data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
          #data.reshape(num_images, image_size, image_size, 1)
          data_torch = torch.from_numpy(data).view(num_imgs, 1, num_rows, num_cols)
          return data_torch
                  

class TinyImagenet(Dataset):
    """
    Defines Tiny Imagenet as for the others pytorch datasets.
    """
    def __init__(self, root, classes=range(200), opt=None, train=True, transform=None,
                 target_transform=None, download=False, label_dict = None, last_features_list=None, 
                 last_feature_labels_list=None, last_model=None, subsample_transform=None, portion_out=0.1, upsample_times=1):
        self.not_aug_transform = transforms.Compose([transforms.ToTensor()])
        self.root = root
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.download = download
        self.label_dict = label_dict

        print("root", root)
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


    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = PIL.Image.fromarray(np.uint8(255 * img))                     ## put it in non transform ????????????
        original_img = img.copy()

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        if self.label_dict is not None:
            target = self.label_dict[str(target)]

        if hasattr(self, 'logits'):
            return img, target, original_img, self.logits[index]

        return img, target
    
    def get_image_class(self, label):

        return self.data[np.array(self.targets) == label]
    
    
def tinyimgnet_c(root, corruption_name, severity, transform=None):
    
    if corruption_name == "train":
        data_folder = root + "/Tiny-ImageNet-C/train"
    else:
        data_folder = root + "/Tiny-ImageNet-C/" + corruption_name + "/" + str(severity)
    
    print(corruption_name)
    dataset = ImageFolder(root=data_folder, transform=transform)
    
    return dataset
        


class cifar10_c(Dataset):
    
    def __init__(self, root, corruption_name, severity, transform=None):
        root = root + "/CIFAR-10-C/"
        data_name = corruption_name + "_" + str(severity)
        with open(root + data_name, "rb") as f:
            self.data, self.label = pickle.load(f)
        self.transform = transform
            
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        
        img = self.data[index] 
        label = self.label[index]
        if self.transform is not None:
            img = PIL.Image.fromarray(img)
            img = self.transform(img)
            
        return img, label


class cifar100_c(Dataset):

    def __init__(self, root, corruption_name, severity, transform=None):
        root = root + "/CIFAR-100-C/"
        data_name = corruption_name + "_" + str(severity)
        with open(root + data_name, "rb") as f:
            self.data, self.label = pickle.load(f)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img = self.data[index]
        label = self.label[index]
        if self.transform is not None:
            img = PIL.Image.fromarray(img)
            img = self.transform(img)

        return img, label



def ImageNet100_Folder(root, classes=range(100), train=True, opt=None, transform=None,
                       target_transform=None, download=False, label_dict = None, last_features_list=None,
                       last_feature_labels_list=None, last_model=None, subsample_transform=None, portion_out=0.1, upsample_times=1):
    
    # put the data in the same directory with the project 
    if train:
        data_path = root + "/imagenet100_train"
    else:
        data_path = root + "/imagenet100_test"
        
    dataset = SafeImageFolder(data_path, transform=transform)

    return dataset



class ImageNet100(Dataset):

    def __init__(self, root, classes=range(100), train=True, opt=None, transform=None,
                target_transform=None, download=False, label_dict = None, last_features_list=None,
                last_feature_labels_list=None, last_model=None, subsample_transform=None, portion_out=0.1, upsample_times=1):

        if train:
            data_path = root + "/imagenet100_train"
        else:
            data_path = root + "/imagenet100_test"

        dataset = SelectImageFolder(data_path, classes)
        self.images = []
        self.labels = []
        self.transform = transform

        for img, l in dataset:
            self.images.append(img)
            self.labels.append(l)

    def __getitem__(self, idx):
        if self.transform is not None:
            return self.transform(self.images[idx]), self.labels[idx]
        else:
            return self.images[idx], self.labels[idx]

    def __len__(self):

        return len(self.images)


class SelectImageFolder(ImageFolder):
    def __init__(self, root, classes=range(100), **kwargs):

        self.target_class_indices = classes
        super().__init__(root, **kwargs)

    def find_classes(self, directory):

        classes_all = [d.name for i, d in enumerate(os.scandir(directory)) if d.is_dir()]
        classes_all.sort()

        classes = [n for i, n in enumerate(classes_all)
                            if i in self.target_class_indices]
        classes_to_idx = {cls_name: i for cls_name, i in zip(classes, self.target_class_indices)}

        return classes, classes_to_idx


class ImageNet900_outliers(Dataset):

    def __init__(self, root, classes=range(100), train=True, opt=None, transform=None,
                target_transform=None, download=False, label_dict = None, last_features_list=None,
                last_feature_labels_list=None, last_model=None, subsample_transform=None, portion_out=0.1, upsample_times=1):

        data_path = root + "/imagenet900_test"

        dataset = ImageFolder(data_path)
        self.images = []
        self.labels = []
        self.transform = transform

        for img, l in dataset:
            self.images.append(img)
            self.labels.append(1000)

    def __getitem__(self, idx):
        if self.transform is not None:
            return self.transform(self.images[idx]), self.labels[idx]
        else:
            return self.images[idx], self.labels[idx]

    def __len__(self):

        return len(self.images)


def ImageNet100_small(root, classes=range(10), train=True, opt=None, transform=None,
                      target_transform=None, download=False, label_dict=None, last_features_list=None,
                      last_feature_labels_list=None, last_model=None, subsample_transform=None, portion_out=0.1,
                      upsample_times=1):
    if train:
        data_path = root + "/imagenet100_small"
    else:
        data_path = root + "/imagenet100_small_test"

    dataset = SafeImageFolder(data_path, transform=transform)

    return dataset


class ImageNet100_M(Dataset):
    
    def __init__(self, root, train, opt = None, classes=range(10), download=True, transform = None, target_transform = None, label_dict = None,
                 last_features_list=None, last_feature_labels_list=None, last_model=None, subsample_transform=None, portion_out=0.1, upsample_times=1):
       if train:
          self.data_path = root + "/imagenet100_train"
          self.annotations_path = root + "/imagenet_annotations/Annotation"
          self.transform = transforms.Compose(
              [transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
               transforms.RandomHorizontalFlip(),
               transforms.RandomGrayscale(p=0.2),
               transforms.ToTensor(),
               transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)), ])
          if "training_supcon" in opt.action:
              self.transform = TwoCropTransform(self.transform)
       else:
          self.data_path = root + "/imagenet100_test"
          self.annotations_path = root + "/imagenet_annotations_vali"
          self.transform = transforms.Compose(
              [transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
               transforms.RandomGrayscale(p=0.2),
               transforms.ToTensor(),
               transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)), ])
          if "testing_known" in opt.action or "testing_unknown" in opt.action:
              self.transform = TwoCropTransform(self.transform)

       self.class_dir_list = sorted(os.listdir(self.data_path))
       self.class_map = {serie_num: index for index, serie_num in enumerate(self.class_dir_list)}
       self.newsize = 224
       self.data = []
       self.targets = []
       self.annotations = []
       self.target_transform = target_transform
       print(self.class_map)

       for idx, cd in enumerate(self.class_dir_list):
           print(idx, cd)
           class_dir = self.data_path + "/" + cd
           class_file_list = os.listdir(class_dir)
           for class_file_name in class_file_list:
               im = PIL.Image.open(class_dir + "/" + class_file_name)
               im = im.convert("RGB")

               self.width, self.height = im.size

               if self.width <= self.newsize or self.height <= self.newsize:
                   im = im.resize((self.newsize, self.newsize))
               else:
                   im = im.crop(((self.width-self.newsize)//2, (self.height-self.newsize)//2, (self.width-self.newsize)//2+self.newsize, (self.height-self.newsize)//2+self.newsize))
               label = self.class_map[cd]
               self.data.append(im)
               self.targets.append(label)

               if train:
                  annotation_name = cd + "_" + class_file_name.split(".")[0].split("_")[-1] + ".xml"
                  annotation_file_path = os.path.join(self.annotations_path, cd, annotation_name)
               else:
                  annotation_name = class_file_name.split(".")[0] + ".xml"
                  annotation_file_path = os.path.join(self.annotations_path, annotation_name)
               if os.path.isfile(annotation_file_path):
                   #print("found annotation")
                   xmin, ymin, xmax, ymax = self.read_annotation(annotation_file_path)
                   #print(xmin, ymin, xmax, ymax)
                   self.annotations.append(torch.tensor([xmin, ymin, xmax, ymax]))

                   """
                   # for testing
                   
                   draw = ImageDraw.Draw(im)
                   triangle_coords = [(xmin, ymin), (xmax, ymax)]
                   draw.rectangle(triangle_coords, fill=None, outline='blue')
                   im.show()
                   """
               else:
                   self.annotations.append(torch.tensor([0, 0, 0, 0]))


    def __len__(self):

       return len(self.data)

    def __getitem__(self, index):

       img, target = self.data[index], self.targets[index]
       annotation = self.annotations[index]

       if self.transform is not None:
            img = self.transform(img)

       if self.target_transform is not None:
            target = self.target_transform(target)

       return  img, target

    def read_annotation(self, annotation_path):
        tree = ET.parse(annotation_path)
        root = tree.getroot()
        bndbox = root.find(".//bndbox")

        xmin = int(bndbox.find("xmin").text)
        ymin = int(bndbox.find("ymin").text)
        xmax = int(bndbox.find("xmax").text)
        ymax = int(bndbox.find("ymax").text)

        #print(xmin, ymin, xmax, ymax)
        xmin, ymin, xmax, ymax = self.convert_annotation(xmin, ymin, xmax, ymax)

        return xmin, ymin, xmax, ymax

    def convert_annotation(self, xmin, ymin, xmax, ymax):

        if self.width <= self.newsize or self.height <= self.newsize:
            xmin = int(xmin * self.newsize / self.width)
            xmax = int(xmax * self.newsize / self.width)
            ymin = int(ymin * self.newsize / self.height)
            ymax = int(ymax * self.newsize / self.height)
        else:
            delta_x = (self.width - self.newsize) / 2
            delta_y = (self.height - self.newsize) / 2
            xmin = int(xmin - delta_x if xmin - delta_x > 0 else 0)
            xmax = int(xmax - delta_x if xmax - delta_x <= self.newsize else self.newsize)
            ymin = int(ymin - delta_y if ymin - delta_y > 0 else 0)
            ymax = int(ymax - delta_y if ymax - delta_y <= self.newsize else self.newsize)

        return  xmin, ymin, xmax, ymax


class customSVHN(SVHN):

    def __init__(self, root, train, opt=None, classes=range(10), download=True, transform = None, target_transform = None, label_dict = None,
                 last_features_list=None, last_feature_labels_list=None, last_model=None, subsample_transform=None, portion_out=0.1, upsample_times=1):
        super(customSVHN, self).__init__(root=root, split=train, transform = transform, target_transform = target_transform, download=download)
    
        self.root = root
        self.train = train
        self.label_dict = label_dict

        if self.split == "train":
            train_data = []
            train_labels = []

            for i in range(len(self.data)):
                if self.labels[i] in classes:
                    train_data.append(self.data[i])
                    train_labels.append(self.labels[i])

            if subsample_transform is not None:
                for idx, (last_features, last_feature_labels) in enumerate(zip(last_features_list, last_feature_labels_list)):
                    kept_indices = datautil.upsample_distance_center_mahalanobis(train_data=train_data, train_labels=train_labels, last_features=last_features, last_feature_labels=last_feature_labels, last_model=last_model,
                                                                                 label_dict=label_dict, num_classes=len(classes), subsample_transform=subsample_transform, portion_out=portion_out, dataset_name="cifar10", 
                                                                                 upsample_times=upsample_times, idx=idx)
                    if idx == 0:
                        final_indices = kept_indices
                    else:
                        final_indices = common_elements(final_indices, kept_indices)
                    
                train_data = [train_data[i] for i in final_indices] 
                train_labels = [train_labels[i] for i in final_indices] 

            self.train_data = np.array(train_data)
            self.train_labels = train_labels

        else:
            test_data = []
            test_labels = []

            for i in range(len(self.data)):
                if self.labels[i] in classes:
                    test_data.append(self.data[i])
                    test_labels.append(self.labels[i])

            self.test_data = np.array(test_data)
            self.test_labels = test_labels


    def __getitem__(self, index):

        if self.split == "train":
            img, target = self.train_data[index], self.train_labels[index]
        else:
            img, target = self.test_data[index], self.test_labels[index]

        if self.transform is not None:
            img = PIL.Image.fromarray(np.transpose(img, (1, 2, 0)))
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        if self.label_dict is not None:
            target = self.label_dict[str(target)]

        return img, target


    def __len__(self):

        if self.split == "train":
            return len(self.train_data)
        else:
            return len(self.test_data)


    def get_image_class(self, label):
        return self.train_data[np.array(self.train_labels) == label]
    
    
    def get_part_data(self, xidxs):
        
        self.train_data = np.delete(self.train_data, xidxs, 0)
        self.train_labels = np.delete(self.train_labels, xidxs, 0)


    def append(self, images, labels):
        """Append dataset with images and labels

        Args:
            images: Tensor of shape (N, C, H, W)
            labels: list of labels
        """

        self.train_data = np.concatenate((self.train_data, images), axis=0)
        self.train_labels = self.train_labels + labels


class CUB(Dataset):

    def __init__(self, root, classes=range(1, 201), opt=None, download=False,
                 train=True, transform=None, target_transform=None,
                 label_dict=None, last_features_list=None, last_features=None, 
                 last_feature_labels=None, last_model=None, last_feature_labels_list=None,
                 subsample_transform=None):
        
        """
        classes should be indexed from 1!!!!
        """
        
        self.root = root
        self.transform = transform
        self.target_trasnform = target_transform
        self.train = train
        self.label_dict = label_dict
        self.loader = default_loader

        images_folder = "/CUB/images/"
        image_list_file = "CUB/images.txt"
        image_class_file = "CUB/image_class_labels.txt"
        train_test_split_file = "CUB/train_test_split.txt"

        with open(os.path.join(self.root, image_list_file), "r") as f:
            lines = f.readlines()

        self.file_dict = {"indices": [], "path": [], "label": [], "train_test_split": []}
        for line in lines:
            index, image_path = line.split(" ")
            index = int(index)
            image_path = image_path[:-1]
            self.file_dict["indices"].append(index)
            self.file_dict["path"].append(image_path)

        with open(os.path.join(self.root, image_class_file), "r") as f:
            lines = f.readlines()

        for i, line in enumerate(lines):
            index, class_index = line.split(" ")
            index = int(index)
            class_index = int(class_index)
            assert i+1 == self.file_dict["indices"][i]
            self.file_dict["label"].append(class_index)

        with open(os.path.join(self.root, train_test_split_file), "r") as f:
            lines = f.readlines()

        for i, line in enumerate(lines):
            index, split = line.split(" ")
            index = int(index)
            split = int(bool(i%5))  #int(split)
            assert i+1 == self.file_dict["indices"][i]
            self.file_dict["train_test_split"].append(split)

        if self.train:
            self.train_data = []
            self.train_labels = []
            for i, (path, label, split) in enumerate(zip(self.file_dict["path"], self.file_dict["label"], self.file_dict["train_test_split"])):
                if split == 0 or label not in classes:
                    continue
                img = self.loader(self.root + images_folder + path)
                self.train_data.append(img)
                self.train_labels.append(label)
                                                                                        
        else:
            self.test_data = []
            self.test_labels = []
            for i, (path, label, split) in enumerate(zip(self.file_dict["path"], self.file_dict["label"], self.file_dict["train_test_split"])):
                if split == 1 or label not in classes:
                    continue
                img = self.loader(self.root + images_folder + path)
                self.test_data.append(img)
                self.test_labels.append(label)


    def __getitem__(self, index):

        if self.train:
            img, target = self.train_data[index], self.train_labels[index]
        else:
            img, target = self.test_data[index], self.test_labels[index]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_trasnform is not None:
            target = self.target_trasnform(target)

        if self.label_dict is not None:
            target = self.label_dict[str(target)]

        return img, target

    
    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)

    
    def get_image_class(self, label):
        return self.train_data[np.array(self.train_labels) == label]
    
    
    def get_part_data(self, xidxs):
        
        self.train_data = np.delete(self.train_data, xidxs, 0)
        self.train_labels = np.delete(self.train_labels, xidxs, 0)


class Aircraft(VisionDataset):
    """`FGVC-Aircraft <http://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/>`_ Dataset.

    Args:
        root (string): Root directory of the dataset.
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        class_type (string, optional): choose from ('variant', 'family', 'manufacturer').
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
    """
    url = 'http://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/archives/fgvc-aircraft-2013b.tar.gz'
    class_types = ('variant', 'family', 'manufacturer')
    splits = ('train', 'val', 'trainval', 'test')
    img_folder = os.path.join('Aircraft', 'data', 'images')

    def __init__(self, root, train=True, opt=None, classes=range(100), download=False,  transform=None,
                 target_transform=None, label_dict = None, last_features_list=None, last_feature_labels_list=None,
                 last_model=None, subsample_transform=None, class_type='variant'):

        super(Aircraft, self).__init__(root, transform=transform, target_transform=target_transform)
        split = 'trainval' if train else 'test'
        if split not in self.splits:
            raise ValueError('Split "{}" not found. Valid splits are: {}'.format(
                split, ', '.join(self.splits),
            ))
        if class_type not in self.class_types:
            raise ValueError('Class type "{}" not found. Valid class types are: {}'.format(
                class_type, ', '.join(self.class_types),
            ))

        self.root = root
        self.class_type = class_type
        self.classes = classes
        self.split = split
        self.loader = default_loader
        self.transform = transform
        self.classes_file = os.path.join(self.root, 'Aircraft', 'data',
                                         'images_%s_%s.txt' % (self.class_type, self.split))

        (image_ids, targets, total_classes, class_to_idx) = self.find_classes()
        samples, targets = self.make_dataset(image_ids, targets)

        self.samples = samples
        self.targets = targets
        self.total_classes = total_classes
        self.class_to_idx = class_to_idx
        self.label_dict = label_dict


    def __getitem__(self, index):
        sample, target = self.samples[index], self.targets[index]
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        if self.label_dict is not None:
            target = self.label_dict[str(target)]
        return sample, target

    def __len__(self):
        return len(self.samples)

    def _check_exists(self):
        return os.path.exists(os.path.join(self.root, self.img_folder)) and \
               os.path.exists(self.classes_file)

    def find_classes(self):
        # read classes file, separating out image IDs and class names
        image_ids = []
        targets = []
        with open(self.classes_file, 'r') as f:
            for line in f:
                split_line = line.split(' ')
                image_ids.append(split_line[0])
                targets.append(' '.join(split_line[1:]))

        # index class names
        classes = np.unique(targets)
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        targets = [class_to_idx[c] for c in targets]

        return image_ids, targets, classes, class_to_idx

    def make_dataset(self, image_ids, targets):
        assert (len(image_ids) == len(targets))
        images = []
        labels = []
        for i in range(len(image_ids)):
            if targets[i] not in self.classes:
                continue
            item = os.path.join(self.root, self.img_folder, '%s.jpg' % image_ids[i])
            image = self.loader(item)
            images.append(image)
            labels.append(targets[i])
        return images, labels


class Cars(Dataset):

    def __init__(self, root, classes=range(100), train=True, opt=None, transform=None,
                target_transform=None, download=False, label_dict = None, last_features_list=None,
                last_feature_labels_list=None, last_model=None, subsample_transform=None, portion_out=0.1, upsample_times=1):

        if train:
            data_dir = os.path.join(root, "Cars/train")
        else:
            data_dir = os.path.join(root, "Cars/test")

        self.dataset = ImageFolder(data_dir)
        self.samples = []
        self.targets = []
        self.transform = transform
        self.target_transform = target_transform
        self.label_dict = label_dict

        for img, l in self.dataset:
            if l in classes:
                self.samples.append(img)
                self.targets.append(l)
        print(len(self.samples))


    def __getitem__(self, index):
        sample, target = self.samples[index], self.targets[index]
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        if self.label_dict is not None:
            target = self.label_dict[str(target)]
        return sample, target


    def __len__(self):

        return len(self.samples)


class FUB(Dataset):

    def __init__(self, root, classes=range(3), train=True, opt=None, transform=None,
                target_transform=None, download=False, label_dict = None, last_features_list=None,
                last_feature_labels_list=None, last_model=None, subsample_transform=None, portion_out=0.1, upsample_times=1):

        self.labels_dict = {"Trans-thalamic": 0, "Trans-cerebellum": 1, "Trans-ventricular": 2, "Other": 3}
        images_folder = os.path.join(root, "fetal-ultrasound-brain/data")
        self.image_names = sorted(os.listdir(images_folder))
        self.labeling_file = os.path.join(root, "fetal-ultrasound-brain/sample_submission.csv")
        self.names_labels_dict = {}
        self.transform = transform

        self.data = []
        self.labels = []

        with open(self.labeling_file, newline="") as f:
            reader = csv.reader(f, delimiter=' ', quotechar='|')
            next(reader)
            for row in reader:
                self.names_labels_dict[row[0].split(",")[0]] = row[0].split(",")[1]

        for fn in self.image_names:
            data_name = fn.split(".")[0]
            label_name = self.names_labels_dict[data_name]
            label = self.labels_dict[label_name]
            if label in classes:
                image = PIL.Image.open(os.path.join(images_folder, fn)).convert('L')
                self.labels.append(label)
                self.data.append(image)
                # upsampling
                if label in [1,2]:
                    self.labels.append(label)
                    self.data.append(image)

        if train:
            del self.data[4::5]
            del self.labels[4::5]
        else:
            self.data = self.data[::5]
            self.labels = self.labels[::5]

    def __getitem__(self, idx):

        if self.transform is not None:
            return self.transform(self.data[idx]), self.labels[idx]
        else:
            return self.data[idx], self.labels[idx]

    def __len__(self):

        return len(self.data)

"""
def ImageNet1K(root, classes=range(100), train=True, opt=None, transform=None,
               target_transform=None, download=False, label_dict = None, last_features_list=None,
               last_feature_labels_list=None, last_model=None, subsample_transform=None, portion_out=0.1, upsample_times=1):
    if train:
        data_dir = os.path.join(root, "imagenet1k")
    else:
        data_dir = os.path.join(root, "imagenet1k_test")

    imagenet1k = SafeImageFolder(data_dir, transform=transform)
    return imagenet1k
"""


def MITScene(root, classes=range(100), train=True, opt=None, transform=None,
            target_transform=None, download=False, label_dict = None, last_features_list=None,
            last_feature_labels_list=None, last_model=None, subsample_transform=None, portion_out=0.1, upsample_times=1):
    if train:
        data_path = root + "/" + "MIT_train"
    else:
        data_path = root + "/" + "MIT_test"

    dataset = ImageFolder(data_path, transform=transform)

    return dataset


def Flower102(root, classes=range(10), train=True, opt=None, transform=None,
              target_transform=None, download=False, label_dict = None, last_features_list=None,
              last_feature_labels_list=None, last_model=None, subsample_transform=None, portion_out=0.1, upsample_times=1):
    if train:
        data_path = root + "/" + "flower_train"
    else:
        data_path = root + "/" + "flower_test"

    dataset = ImageFolder(data_path, transform=transform)

    return dataset


def DTD(root, classes=range(100), train=True, opt=None, transform=None,
        target_transform=None, download=False, label_dict = None, last_features_list=None,
        last_feature_labels_list=None, last_model=None, subsample_transform=None, portion_out=0.1, upsample_times=1):
    if train:
        data_path = root + "/" + "DTD_train"
    else:
        data_path = root + "/" + "DTD_test"

    dataset = ImageFolder(data_path, transform=transform)

    return dataset


class ImageNet1K(Dataset):

    def __init__(self, root, classes=range(1000), train=True, opt=None, transform=None,
                target_transform=None, download=False, label_dict = None, last_features_list=None,
                last_feature_labels_list=None, last_model=None, subsample_transform=None, portion_out=0.1, upsample_times=1):

        if train:
            data_path = root + "/imagenet1k"
        else:
            data_path = root + "/imagenet1k_test"

        dataset = SelectImageFolder(data_path, classes)
        self.images = []
        self.labels = []
        self.transform = transform

        for i in range(len(dataset)):
            try:
                img, l = dataset[i]
                self.images.append(img)
                self.labels.append(l)
            except (PIL.UnidentifiedImageError, OSError, IOError) as e:
                continue

    def __getitem__(self, idx):
        """
        try:
            return self.transform(self.images[idx]), self.labels[idx]
        except (PIL.UnidentifiedImageError, OSError, IOError) as e:
            # Log warning (optional)
            path, _ = self.images[idx]
            print(f"[Warning] Failed to read {path}: {e}")
            return None
        """
        if self.transform is not None:
            return self.transform(self.images[idx]), self.labels[idx]
        else:
            return self.images[idx], self.labels[idx]


    def __len__(self):

        return len(self.images)


class SafeImageFolder(ImageFolder):

  def __getitem__(self, index):
    try:
      return super().__getitem__(index)
    except (PIL.UnidentifiedImageError, OSError, IOError) as e:
      # Log warning (optional)
      path, _ = self.samples[index]
      print(f"[Warning] Failed to read {path}: {e}")
      return None


# 2. Custom collate function to drop None items from the batch
def safe_collate(batch):
  batch = [item for item in batch if item is not None]
  if len(batch) == 0:
    return None  # Handle empty batch in your training loop
  return default_collate(batch)


def imagenet_o(transform=None, target_transform=None, download=False, label_dict = None, last_features_list=None,):
    def collate_fn(batch):
        pixel_values = torch.stack([x["pixel_values"] for x in batch])
        return pixel_values

    def transform_fn(examples):
        # Convert PIL Images to PyTorch Tensors and apply transforms
        examples["pixel_values"] = [transform(img.convert("RGB")) for img in examples["image"]]
        return examples

    dataset = load_dataset("Voxel51/ImageNet-O", split="train")
    dataset.set_transform(transform_fn)

    return dataset



if __name__ == "__main__":
    
    root_path = "../datasets/"

    """
    transform = transforms.Compose([transforms.ToTensor(), transforms.CenterCrop((224, 288)),
                                    transforms.Resize(224),
                                    transforms.RandomHorizontalFlip(), transforms.RandomRotation(15),
                                    transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
                                    transforms.RandomGrayscale(p=0.2),
                                    ])

    root = "../datasets/"
    fub0 = FUB(root=root, classes=[0], transform=transform, train=True)
    fub1 = FUB(root=root, classes=[1], transform=transform, train=True)
    fub2 = FUB(root=root, classes=[2], transform=transform, train=True)
    train_loader = torch.utils.data.DataLoader(fub0, batch_size=32, shuffle=True,
                                               num_workers=1, pin_memory=True,
                                               drop_last=True,
                                               persistent_workers=True)
    print(len(fub0))
    print(len(fub1))
    print(len(fub2))
    """

    transform = transforms.Compose([transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
                                          transforms.RandomResizedCrop(size=224, scale=(0.2, 1.)),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.RandomGrayscale(p=0.2),
                                          transforms.ToTensor(),])

    imagenet100 = ImageNet100(root=root_path, classes=[0,1,2,3,4], train=False, transform=transform)
    for i, (_, l) in enumerate(imagenet100):
        print(i, l)

    """
    cub = CUB(root_path)
    print("CUB", len(cub))

    cars = Cars(root_path)
    print("Cars", len(cars))

    aircraft = Aircraft(root_path)
    print("Aircraft", len(aircraft))
    """


   
