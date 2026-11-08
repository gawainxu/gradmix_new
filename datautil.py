#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 25 21:42:16 2021

@author: zhi
"""

classMap = {0: "apples", 1: "aquarium_fish", 2: "baby", 3: "bear", 4: "beaver",
            5: "bed", 6: "bee", 7: "beetle", 8: "bicycle", 9: "bottles",
            10: "bowls", 11: "boy", 12: "bridge", 13: "bus", 14: "butterfly",
            15: "camel", 16: "cans", 17: "castle", 18: "caterpillar", 19: "cattle",
            20: "chair", 21: "chimpanzee", 22: "clock", 23: "cloud", 24: "cockroach",
            25: "couch", 26: "crab", 27: "crocodile", 28: "cups", 29: "dinosaur",
            30: "dolphin", 31: "elephant", 32: "flatfish", 33: "forest", 34: "fox",
            35: "girl", 36: "hamster", 37: "house", 38: "kangaroo", 39: "keyboard", 
            40: "lamp", 41: "lawn_mower", 42: "leopard", 43: "lion", 44: "lizard",
            45: "lobster", 46: "man", 47: "maple", 48: "motorcycle", 49: "mountain",
            50: "mouse", 51: "mushrooms", 52: "oak", 53: "oranges", 54: "orchids", 
            55: "otter", 56: "palm", 57: "pears", 58: "pickup_truck", 59: "pine",
            60: "plain", 61: "plates", 62: "poppies", 63: "porcupine", 64: "possum",
            65: "rabbit", 66: "raccoon", 67: "ray", 68: "road", 69: "rocket",
            70: "roses", 71: "sea", 72: "seal", 73: "shark", 74: "shrew",
            75: "skunk", 76: "skyscraper", 77: "snail", 78: "snake", 79: "spider",
            80: "squirrel", 81: "streetcar", 82: "sunflowers", 83: "pepper", 84: "table", 
            85: "tank", 86: "telephone", 87: "television", 88: "tiger", 89: "tractor",
            90: "train", 91: "trout", 92: "tulips", 93: "turtle", 94: "wardrobe",
            95: "whale", 96: "willow", 97: "wolf", 98: "woman", 99: "worm"}

classMap = {v : k for k, v in classMap.items()}

superClasses = [["beaver", "dolphin", "otter", "seal", "whale"],
                ["aquarium_fish", "flatfish", "ray", "shark", "trout"],
                ["orchids", "poppies", "roses", "sunflowers", "tulips"],
                ["bottles", "bowls", "cans", "cups", "plates"],
                ["apples", "mushrooms", "oranges", "pears", "peppers"],
                ["clock", "keyboard", "lamp", "telephone", "television"],
                ["bed", "chair", "couch", "table", "wardrobe"],
                ["bee", "beetle", "butterfly", "caterpillar", "cockroach"],
                ["bear", "leopard", "lion", "tiger", "wolf"],
                ["bridge", "castle", "house", "road", "skyscraper"],
                ["cloud", "forest", "mountain", "plain", "sea"],
                ["camel", "cattle", "chimpanzee", "elephant", "kangaroo"],
                ["fox", "porcupine", "possum", "raccoon", "skunk"],
                ["crab", "lobster", "snail", "spider", "worm"],
                ["baby", "boy", "girl", "man", "woman"],
                ["crocodile", "dinosaur", "lizard", "snake", "turtle"],
                ["hamster", "mouse", "rabbit", "shrew", "squirrel"],
                ["maple", "oak", "palm", "pine", "willow"],
                ["bicycle", "bus", "motorcycle", "pickup_truck", "train"],
                ["lawn_mower", "rocket", "streetcar", "tank", "tractor"]]


osr_splits_inliers = {
    # Training classes
    'mnist': [
        [2, 4, 5, 9, 8, 3],
        [3, 2, 6, 9, 4, 0],
        [5, 8, 3, 2, 4, 6],
        [3, 7, 8, 4, 0, 5],
        [6, 3, 4, 9, 8, 2]
    ],
    'svhn': [
        [5, 3, 7, 2, 8, 6],
        [3, 8, 7, 6, 2, 5],
        [8, 9, 4, 7, 2, 1],
        [3, 8, 2, 5, 0, 6],
        [4, 9, 2, 7, 1, 0],
    ],

    'cifar10': [
        [0, 6, 4, 9, 1, 7],                    
        [7, 6, 4, 9, 0, 1],
        [1, 5, 7, 3, 9, 4],
        [8, 6, 1, 9, 0, 7],
        [2, 4, 1, 7, 9, 6],
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    ],

    'cifar-10-100-10': [
        [4, 7, 9, 1],
        [6, 7, 1, 9],
        [9, 6, 1, 7],
        [6, 4, 9, 1],
        [1, 0, 9, 8]
    ],

    'cifar-10-100-50': [
        [4, 7, 9, 1],
        [6, 7, 1, 9],
        [9, 6, 1, 7],
        [6, 4, 9, 1],
        [1, 0, 9, 8]
    ],

    "cifar100": [list(range(100))],
    
    "imagenet100": [list(range(100))],

    "imagenet100_small": [list(range(10))],

    "imagenet100_m": [list(range(100))],

    "ImageNet100_Folder": [list(range(100))],

     "imagenet1k": [list(range(1000))],

    'tinyimgnet': [
        [108, 147, 17, 58, 193, 123, 72, 144, 75, 167, 134, 14, 81, 171, 44, 197, 152, 66, 1, 133],
        [198, 161, 91, 59, 57, 134, 61, 184, 90, 35, 29, 23, 199, 38, 133, 19, 186, 18, 85, 67],
        [177, 0, 119, 26, 78, 80, 191, 46, 134, 92, 31, 152, 27, 60, 114, 50, 51, 133, 162, 93],
        [98, 36, 158, 177, 189, 157, 170, 191, 82, 196, 138, 166, 43, 13, 152, 11, 75, 174, 193, 190],
        [95, 6, 145, 153, 0, 143, 31, 23, 189, 81, 20, 21, 89, 26, 36, 170, 102, 177, 108, 169],
        list(range(200)),
        list(range(100))
    ],
    
    "cub": [[151, 71, 35, 179, 200, 132, 130, 148, 135, 12, 27, 94, 96, 122, 124, 100, 150, 168, 19, 32, 70, 
             199, 117, 159, 127, 18, 6, 180, 112, 164, 185, 82, 175, 43, 54, 90, 78, 56, 24, 49, 44, 45, 57, 
             29, 194, 144, 1, 177, 85, 16, 39, 155, 142, 191, 173, 125, 190, 20, 81, 158, 13, 10, 80, 31, 95, 
             68, 198, 98, 169, 138, 120, 77, 99, 89, 41, 107, 172, 88, 167, 187, 28, 52, 145, 136, 162, 65, 178,
             8, 147, 62, 51, 163, 134, 83, 40, 75, 73, 92, 197, 137]], 
            
    "aircraft": [[0, 1, 2, 3, 4, 5, 10, 11, 14, 16, 17, 19, 21, 22, 23, 24, 27, 28, 29, 30, 33,
                  36, 37, 38, 39, 41, 43, 44, 45, 46, 47, 48, 52, 53, 56, 57, 58, 63, 64, 65,
                  66, 67, 71, 73, 76, 77, 79, 92, 95, 99]],

    "cars": [[1, 11, 25, 38, 46, 50, 53, 75, 84, 100, 105, 117, 123, 129, 133, 134, 135, 136,
                      137, 138, 140, 144, 145, 146, 147, 149, 150, 151, 153, 160, 161, 162, 163, 164,
                      167, 168, 169, 174, 175, 180, 185, 186, 187, 192, 193, 0, 81, 97, 104, 122, 139,
                      141, 142, 143, 148, 152, 154, 155, 156, 157, 158, 159, 165, 166, 170, 171, 172,
                      173, 176, 177, 181, 184, 188, 191, 194, 195, 2, 7, 9, 16, 20, 26, 28, 44, 54, 95,
                      98, 102, 127, 178, 182, 22, 41, 82, 93, 112, 125, 189]],

    "mit": [list(range(67))],

    "dtd": [list(range(47))],

    "flower": [list(range(102))],
                  
    "cifar100_macro": [[4, 54, 3, 22, 26, 8]],

    "FUB": [[0, 1, 2]]}


osr_splits_outliers = {
    # testing classes
    "mnist": [[0, 1, 6, 7],
              [1, 5, 7, 8],
              [0, 1, 7, 9],
              [1, 2, 6, 9],
              [0, 1, 5, 7]],

    "svhn": [[0, 1, 4, 9],
             [0, 1, 4, 9],
             [0, 3, 5, 6],
             [1, 4, 7, 9],
             [3, 5, 6, 8]],

    "cifar10": [[2, 3, 5, 8],
                [2, 3, 5, 8],
                [0, 2, 6, 8],
                [2, 3, 4, 5],
                [0, 3, 5, 8],
                [0]],

    "cifar100": [[0]],

    'cifar-10-100-10': [
        [30, 25, 1, 9, 8, 0, 46, 52, 49, 71],
        [41, 9, 49, 40, 73, 60, 48, 30, 95, 71],
        [8, 9, 49, 40, 73, 60, 48, 95, 30, 71],
        [95, 60, 30, 73, 46, 49, 68, 99, 8, 71],
        [33, 2, 3, 97, 46, 21, 64, 63, 88, 43]
    ],

    'cifar-10-100-50': [
        [27, 94, 29, 77, 88, 26, 69, 48, 75, 5, 59, 93, 39, 57, 45, 40, 78, 20, 98, 47, 66, 70, 91, 76, 41, 83, 99, 32, 53, 72, 2, 95, 21, 73, 84, 68, 35, 11, 55, 60, 30, 25, 1, 9, 8, 0, 46, 52, 49, 71],
        [65, 97, 86, 24, 45, 67, 2, 3, 91, 98, 79, 29, 62, 82, 33, 76, 0, 35, 5, 16, 54, 11, 99, 52, 85, 1, 25, 66, 28, 84, 23, 56, 75, 46, 21, 72, 55, 68, 8, 69, 41, 9, 49, 40, 73, 60, 48, 30, 95, 71],
        [20, 83, 65, 97, 94, 2, 93, 16, 67, 29, 62, 33, 24, 98, 5, 86, 35, 54, 0, 91, 52, 66, 85, 84, 56, 11, 1, 76, 25, 55, 21, 99, 72, 41, 23, 75, 28, 68, 69, 46, 8, 9, 49, 40, 73, 60, 48, 95, 30, 71],
        [92, 82, 77, 64, 5, 33, 62, 56, 70, 0, 20, 28, 67, 14, 84, 53, 91, 29, 85, 2, 52, 83, 75, 35, 11, 21, 72, 98, 55, 1, 41, 76, 25, 66, 69, 9, 48, 54, 40, 23, 95, 60, 30, 73, 46, 49, 68, 99, 8, 71],
        [47, 6, 19, 0, 62, 93, 59, 65, 54, 70, 34, 55, 23, 38, 72, 76, 53, 31, 78, 96, 77, 27, 92, 18, 82, 50, 98, 32, 1, 75, 83, 4, 51, 35, 80, 11, 74, 66, 36, 42, 33, 2, 3, 97, 46, 21, 64, 63, 88, 43]
    ],

    'tinyimgnet': [  [0,   2,   3,   4,   5,   6,   7,   8,   9,  10,  11,  12,  13, 15,  16,  18,  19,  20,  21,  22,  23,  24,  25,  26,  27,  28, 29,  30,  31,  32,  33,  34,  35,  36,  37,  38,  39,  40,  41,
                      42,  43,  45,  46,  47,  48,  49,  50,  51,  52,  53,  54,  55, 56,  57,  59,  60,  61,  62,  63,  64,  65,  67,  68,  69,  70, 71,  73,  74,  76,  77,  78,  79,  80,  82,  83,  84,  85,  86,
                      87,  88,  89,  90,  91,  92,  93,  94,  95,  96,  97,  98,  99, 100, 101, 102, 103, 104, 105, 106, 107, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 124, 125, 126, 127,
                      128, 129, 130, 131, 132, 135, 136, 137, 138, 139, 140, 141, 142, 143, 145, 146, 148, 149, 150, 151, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 168, 169, 170, 172, 173,
                      174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 194, 195, 196, 198, 199],
                     [ 0,   1,   2,   3,   4,   5,   6,   7,   8,   9,  10,  11,  12, 13,  14,  15,  16,  17,  20,  21,  22,  24,  25,  26,  27,  28, 30,  31,  32,  33,  34,  36,  37,  39,  40,  41,  42,  43,  44,
                       45,  46,  47,  48,  49,  50,  51,  52,  53,  54,  55,  56,  58, 60,  62,  63,  64,  65,  66,  68,  69,  70,  71,  72,  73,  74, 75,  76,  77,  78,  79,  80,  81,  82,  83,  84,  86,  87,  88,
                       89,  92,  93,  94,  95,  96,  97,  98,  99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129,
                       130, 131, 132, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171,
                       172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 185, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197],
                     [1,   2,   3,   4,   5,   6,   7,   8,   9,  10,  11,  12,  13, 14,  15,  16,  17,  18,  19,  20,  21,  22,  23,  24,  25,  28, 29,  30,  32,  33,  34,  35,  36,  37,  38,  39,  40,  41,  42,
                      43,  44,  45,  47,  48,  49,  52,  53,  54,  55,  56,  57,  58, 59,  61,  62,  63,  64,  65,  66,  67,  68,  69,  70,  71,  72, 73,  74,  75,  76,  77,  79,  81,  82,  83,  84,  85,  86,  87,
                      88,  89,  90,  91,  94,  95,  96,  97,  98,  99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 115, 116, 117, 118, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130,
                      131, 132, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 153, 154, 155, 156, 157, 158, 159, 160, 161, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173,
                      174, 175, 176, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 192, 193, 194, 195, 196, 197, 198, 199],
                      [0,   1,   2,   3,   4,   5,   6,   7,   8,   9,  10,  12,  14, 15,  16,  17,  18,  19,  20,  21,  22,  23,  24,  25,  26,  27, 28,  29,  30,  31,  32,  33,  34,  35,  37,  38,  39,  40,  41,
                      42,  44,  45,  46,  47,  48,  49,  50,  51,  52,  53,  54,  55, 56,  57,  58,  59,  60,  61,  62,  63,  64,  65,  66,  67,  68, 69,  70,  71,  72,  73,  74,  76,  77,  78,  79,  80,  81,  83,
                      84,  85,  86,  87,  88,  89,  90,  91,  92,  93,  94,  95,  96, 97,  99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123,
                      124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 153, 154, 155, 156, 159, 160, 161, 162, 163, 164, 165, 167,
                      168, 169, 171, 172, 173, 175, 176, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 192, 194, 195, 197, 198, 199],
                      [ 1,   2,   3,   4,   5,   7,   8,   9,  10,  11,  12,  13,  14, 15,  16,  17,  18,  19,  22,  24,  25,  27,  28,  29,  30,  32, 33,  34,  35,  37,  38,  39,  40,  41,  42,  43,  44,  45,  46,
                        47,  48,  49,  50,  51,  52,  53,  54,  55,  56,  57,  58,  59, 60,  61,  62,  63,  64,  65,  66,  67,  68,  69,  70,  71,  72, 73,  74,  75,  76,  77,  78,  79,  80,  82,  83,  84,  85,  86,
                       87,  88,  90,  91,  92,  93,  94,  96,  97,  98,  99, 100, 101, 103, 104, 105, 106, 107, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129,
                       130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 144, 146, 147, 148, 149, 150, 151, 152, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 171, 172, 173,
                       174, 175, 176, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199],
                     [0], [0]],

    "imagenet100": [[100]],

    "imagenet1k": [list(range(1000))],
    
    "cub": [[21, 160, 174, 149, 2, 58, 114, 166, 53, 110, 15, 5, 181, 7, 183, 69, 
             34, 109, 47, 36, 76, 189, 188, 101, 48, 106, 42, 87, 17, 55, 140, 139], 
            [153, 196, 133, 84, 23, 193, 154, 176, 192, 156, 50, 195, 74, 67, 171, 152, 170, 
             97, 104, 38, 182, 128, 79, 22, 11, 165, 63, 3, 184, 86, 46, 61, 93, 186],
            [30, 111, 4, 9, 14, 59, 143, 26, 146, 64, 60, 66, 25, 141, 121, 33, 115, 108, 161, 131, 119, 
             102, 116, 129, 118, 72, 157, 113, 37, 123, 105, 103, 91, 126]],

    "aircraft": [[6, 7, 8, 9, 12, 13, 31, 32, 25, 26, 18, 20, 15],
                 [78, 82, 51, 49, 50, 54, 55, 59, 60, 61, 68, 69, 70, 85, 86, 87, 88], 
                 [80, 81, 42, 84, 40, 90, 74, 75, 97, 98, 34, 35, 93, 94, 96, 72, 91, 83, 62, 89]],
    
    "cars": [[23, 42, 83, 94,],
                     [3, 8, 10, 17, 21, 27, 29, 45, 55, 96, 99, 103, 128, 179, 183],
                     [4, 5, 6, 12, 13, 14, 15, 18, 19, 24, 30, 31, 32, 33, 34, 35, 36,
                      37, 39, 40, 43, 47, 48, 49, 51, 52, 56, 57, 58, 59, 60, 61, 62,
                      63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 76, 77, 78, 79,
                      80, 85, 86, 87, 88, 89, 90, 91, 92, 101, 106, 107, 108, 109, 110,
                      111, 114, 115, 116, 118, 119, 120, 121, 124, 130, 131, 132]],

    "mit": [list(range(67))],

    "dtd": [list(range(47))],

    "flower": [list(range(102))],

    "cifar100_marco": [[1, 0, 5, 34, 6, 41]],

    "FUB": [[3]]
}


def pickClass(classIdx):
    
    classNames = superClasses[classIdx]
    classList = []
    for n in classNames:
        classList.append(classMap[n])
        
    return classList


import cv2
import random
from data_loader import iCIFAR10, iCIFAR100, TinyImagenet, customSVHN, mnist, CUB, Aircraft, Cars, ImageNet100_M, ImageNet900_outliers
from data_loader import tinyimgnet_c, cifar10_c, cifar100_c, ImageNet100, ImageNet100_small, ImageNet100_Folder, FUB, ImageNet1K
from data_loader import MITScene, DTD, Flower102
import torchvision
import torch
from util import TwoCropTransform
from torchvision import transforms, datasets
import torch.nn.functional as F
from config import data_root
from augmentations.randaugment import RandAugment
from augmentations.cut_out import *
from scipy.spatial.distance import mahalanobis
from util import feature_stats
from PIL import Image
from util import accuracy_plain
from losses import SupConLoss
from gradient_cache import gradient_cache_activations

from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.cluster import KMeans


num_inlier_classes_mapping = {"cifar10": 6, "cifar-10-100-10": 4, "cifar-10-100-50": 4, "cifar100_marco": 6, "imagenet100": 100,
                               "imagenet100_small": 10, "imagenet100_m": 100, "ImageNet100_Folder": 100, "cifar100": 100, "tinyimgnet": 20,
                              "imagenet1k": 1000, "mnist": 6, "svhn": 6, "cub": 100, "aircraft": 50, "cars": 98, "FUB": 3}


data_function_mapping = {"cifar10": iCIFAR10, "cifar-10-100-10": iCIFAR10, "cifar-10-100-50": iCIFAR10, "cifar100_marco": iCIFAR100, "imagenet100": ImageNet100,
                         "imagenet100_small": ImageNet100_small, "imagenet100_m": ImageNet100_M, "ImageNet100_Folder": ImageNet100_Folder, "cifar100": iCIFAR100,
                         "tinyimgnet": TinyImagenet,  "imagenet1k": ImageNet1K, "mnist": mnist, "svhn": customSVHN, "cub": CUB, "aircraft": Aircraft, "cars": Cars,
                         "FUB": FUB, "mit": MITScene, "flower": Flower102, "dtd": DTD}

data_function_mapping_testing = {"cifar10": iCIFAR10, "cifar-10-100-10": iCIFAR100, "cifar-10-100-50": iCIFAR100, "cifar100_marco": iCIFAR100, "imagenet100": ImageNet100,
                                 "imagenet100_small": ImageNet100_small, "imagenet100_m": ImageNet100_M, "ImageNet100_Folder": ImageNet100_Folder, "cifar100": iCIFAR100,
                                 "tinyimgnet": TinyImagenet, "imagenet1k": ImageNet1K, "mnist": mnist, "svhn": customSVHN, "cub": CUB, "aircraft": Aircraft, "cars": Cars,
                                 "FUB": FUB, "mit": MITScene, "flower": Flower102, "dtd": DTD}

data_function_mapping_outliers = {"cifar10": iCIFAR10, "cifar-10-100-10": iCIFAR100, "cifar-10-100-50": iCIFAR100, "cifar100_marco": iCIFAR100, "imagenet100": ImageNet900_outliers,
                                 "imagenet100_small": ImageNet100_small, "imagenet100_m": ImageNet900_outliers, "ImageNet100_Folder": ImageNet900_outliers, "cifar100": iCIFAR100,
                                 "tinyimgnet": TinyImagenet,  "imagenet1k": ImageNet1K, "mnist": mnist, "svhn": customSVHN, "cub": CUB, "aircraft": Aircraft, "cars": Cars, "FUB": FUB,
                                  "mit": MITScene, "flower": Flower102, "dtd": DTD}

data_function_mapping_curruption = {"cifar10": cifar10_c, "cifar100": cifar100_c, "tinyimgnet": tinyimgnet_c,}


mean_mapping = {"mnist":  (0.1307,),
                "svhn": (0.4376821, 0.4437697, 0.47280442),
                "cifar10": (0.4914, 0.4822, 0.4465),
                "cifar100": (0.4914, 0.4822, 0.4465),
                "cifar100_marco": (0.4914, 0.4822, 0.4465),
                "cifar-10-100-10": (0.4914, 0.4822, 0.4465),
                "cifar-10-100-50": (0.4914, 0.4822, 0.4465),
                "imagenet100": (0.485, 0.456, 0.406),
                "ImageNet100_Folder": (0.485, 0.456, 0.406),
                "imagenet100_small": (0.485, 0.456, 0.406),
                "imagenet100_m": (0.485, 0.456, 0.406),
                "imagenet1k": (0.485, 0.456, 0.406),
                "tinyimgnet": (0.485, 0.456, 0.406),
                "aircraft": (0.485, 0.456, 0.406), 
                "cub": (0.485, 0.456, 0.406),
                "cars": (0.485, 0.456, 0.406),
                "FUB": (0.485, 0.456, 0.406),
                "mit": (0.485, 0.456, 0.406),
                "flower": (0.485, 0.456, 0.406),
                "dtd": (0.485, 0.456, 0.406),}       # 0.408, 0.459, 0.502, 123., 117., 104.

std_mapping = {"mnist": (0.3081,),
               "svhn": (0.19803012, 0.20101562, 0.19703614),
               "cifar10": (0.2023, 0.1994, 0.2010),
               "cifar100": (0.2023, 0.1994, 0.2010),
               "cifar100_marco": (0.2023, 0.1994, 0.2010),
               "cifar-10-100-10": (0.2023, 0.1994, 0.2010),
               "cifar-10-100-50": (0.2023, 0.1994, 0.2010),
               "imagenet100": (0.229, 0.224, 0.225),
               "imagenet100_small": (0.229, 0.224, 0.225),
               "ImageNet100_Folder": (0.229, 0.224, 0.225),
               "imagenet100_m": (0.229, 0.224, 0.225),
               "imagenet1k": (0.229, 0.224, 0.225),
               "tinyimgnet": (0.229, 0.224, 0.225),
               "aircraft": (0.229, 0.224, 0.225),
               "cub": (0.229, 0.224, 0.225),
               "cars": (0.229, 0.224, 0.225),
               "FUB": (0.229, 0.224, 0.225),
               "mit": (0.229, 0.224, 0.225),
               "flower": (0.229, 0.224, 0.225),
               "dtd": (0.229, 0.224, 0.225),}

image_size_mapping = {"mnist": 32,
                      "svhn": 32,
                      "cifar10": 32,
                      "cifar100": 32,
                      "cifar100_marco": 32,
                      "cifar-10-100-10": 32,
                      "cifar-10-100-50": 32,
                      "tinyimgnet": 64, 
                      "imagenet100": 224,
                      "imagenet100_small": 224,
                      "ImageNet100_Folder": 224,
                      "imagenet100_m": 224,
                      "imagenet1k": 224,
                      "aircraft": 224,
                      "cub": 224,
                      "cars": 224,
                      "FUB": 128,
                      "mit": 224,
                      "flower": 224,
                      "dtd": 224}


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


def get_train_datasets(opt, class_idx=None, last_features_list=None, last_feature_labels_list=None, last_model=None):

    mean = mean_mapping[opt.datasets]
    std = std_mapping[opt.datasets]
    normalize = transforms.Normalize(mean=mean, std=std)
    size = image_size_mapping[opt.datasets]

    if opt.action == "training_supcon" or opt.action == "trainging_linear":
        if opt.datasets == "mnist":
            train_transform = transforms.Compose([transforms.ToTensor(), transforms.RandomRotation((-5, 5)),])
        elif opt.datasets == "FUB":
            train_transform = transforms.Compose([transforms.ToTensor(), transforms.CenterCrop((224, 288)),
                                    transforms.Resize((size)),
                                    transforms.RandomHorizontalFlip(), transforms.RandomRotation(15),
                                    transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
                                    transforms.RandomGrayscale(p=0.2),])
        elif opt.datasets in ["imagenet100", "imagenet100_small", "ImageNet100_Folder", "imagenet100_m", "cub", "cars", "aircraft", "imagenet1k", "mit", "flower", "dtd"]:
            train_transform = transforms.Compose(
                [transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
                 transforms.Resize((size, size)),
                 transforms.RandomHorizontalFlip(),
                 transforms.RandomGrayscale(p=0.2),
                 # transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),    # !!!!!!!!!!!
                 transforms.ToTensor(),
                 normalize, ])
        else:
            train_transform = transforms.Compose([transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),                                     
                                                  #transforms.Resize((size, size)),
                                                  transforms.RandomResizedCrop(size=size, scale=(0.2, 1.)),    #!!!!!!
                                                  transforms.RandomHorizontalFlip(),
                                                  transforms.RandomGrayscale(p=0.2),
                                                  #transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),    # !!!!!!!!!!!
                                                  transforms.ToTensor(),
                                                  normalize,])   # normalize,
        if opt.randaug == 1:
            train_transform.transforms.insert(0, RandAugment(args=opt))
            
        if opt.augmix:
            # all are default settings
            train_transform.transforms.insert(0, transforms.AugMix())
                
    else:
        if opt.datasets == "mnist":
            train_transform = transforms.Compose([transforms.ToTensor()])
        elif opt.datasets == "FUB":
            train_transform = transforms.Compose([transforms.ToTensor(), transforms.CenterCrop((224, 288)),
                                                  transforms.Resize((size, size))])
        elif opt.datasets in ["imagenet100", "imagenet100_small", "ImageNet100_Folder", "imagenet100_m",
                              "cub", "cars", "aircraft", "mit", "dtd", "flower", "imagenet1k"]:
            train_transform = transforms.Compose([transforms.ToTensor(),
                                                  transforms.Resize((size, size)), normalize])
        else:
            train_transform = transforms.Compose([transforms.ToTensor(), normalize])

    if last_features_list is not None and last_feature_labels_list is not None:
          subsample_transform = transforms.Compose([transforms.RandomResizedCrop(size=size, ),
                                                    transforms.ToTensor(), normalize,])
    else:
          subsample_transform = None

    data_fun = data_function_mapping[opt.datasets]
    label_dict = label_to_dict(osr_splits_inliers[opt.datasets][opt.trail])

    if opt.action == "training_supcon":
        train_transform = TwoCropTransform(train_transform)
    
    if class_idx is not None:
        classes = [osr_splits_inliers[opt.datasets][opt.trail][class_idx]]
    else:
        classes = osr_splits_inliers[opt.datasets][opt.trail]

    if opt.datasets == "svhn":
        train = "train"
    else:
        train = True

    train_dataset = data_fun(root=data_root, train=train, opt=opt,
                             classes=classes, download=False,
                             transform=train_transform, label_dict=label_dict,
                             last_features_list=last_features_list, 
                             last_feature_labels_list=last_feature_labels_list, 
                             last_model=last_model, subsample_transform=subsample_transform, 
                             )        # portion_out=opt.portion_out, upsample_times=opt.upsample_times
    print("dataset size", len(train_dataset))
    return train_dataset


def get_test_datasets(opt, class_idx = None):

    mean = mean_mapping[opt.datasets]
    std = std_mapping[opt.datasets]
    normalize = transforms.Normalize(mean=mean, std=std)
    size = image_size_mapping[opt.datasets]

    if opt.datasets == "mnist":
        test_transform = transforms.Compose([transforms.ToTensor(), ])
    elif opt.datasets == "FUB":
        test_transform = transforms.Compose([transforms.ToTensor(), transforms.CenterCrop((224, 288)),
                                              transforms.CenterCrop((size, size)),])
    elif opt.datasets in ["imagenet100", "imagenet100_small", "ImageNet100_Folder", "imagenet100_m",
                          "cub", "cars", "aircraft", "mit", "dtd", "flower", "imagenet1k"]:
        test_transform = transforms.Compose([transforms.ToTensor(), transforms.Resize((224, 224)), normalize])
    else:
        test_transform = transforms.Compose([transforms.ToTensor(), normalize])

    data_fun = data_function_mapping_testing[opt.datasets]
    label_dict = label_to_dict(osr_splits_inliers[opt.datasets][opt.trail])

    if opt.action == "training_supcon":
        test_transform = TwoCropTransform(test_transform)

    if class_idx is not None:
        classes = [osr_splits_inliers[opt.datasets][opt.trail][class_idx]]
    else:
        classes = osr_splits_inliers[opt.datasets][opt.trail]
    print(classes)
    if opt.datasets == "svhn":
        train = "test"
    else:
        train = False
    test_dataset = data_fun(root=data_root, train=train, opt=opt,
                            classes=classes, download=False,
                            transform=test_transform, label_dict=label_dict)
    print("dataset size", len(test_dataset))
    return test_dataset


def get_curruption_datasets(opt,):

    mean = mean_mapping[opt.datasets]
    std = std_mapping[opt.datasets]
    normalize = transforms.Normalize(mean=mean, std=std)

    test_transform = transforms.Compose([transforms.ToTensor(), normalize])

    data_fun = data_function_mapping_curruption[opt.datasets]

    test_dataset = data_fun(root=data_root, corruption_name=opt.corruption_name, severity=opt.severity, transform=test_transform)
    print("dataset size", len(test_dataset))
    return test_dataset


def get_gradcam_datasets(opt, class_idx = None, transform=None):

    data_fun = data_function_mapping[opt.datasets]
    label_dict = label_to_dict(osr_splits_inliers[opt.datasets][opt.trail])

    if class_idx is not None:
        classes = [osr_splits_inliers[opt.datasets][opt.trail][class_idx]]
    else:
        classes = osr_splits_inliers[opt.datasets][opt.trail]

    transform = TwoCropTransform(transform, origin=True)
    gradcam_dataset = data_fun(root=data_root, train="train",
                               classes=classes, download=True, 
                               transform=transform, label_dict=label_dict)
    print("dataset size", len(gradcam_dataset))
    return gradcam_dataset


def get_outlier_datasets(opt, class_idx=None):

    mean = mean_mapping[opt.datasets]
    std = std_mapping[opt.datasets]
    normalize = transforms.Normalize(mean=mean, std=std)
    size = image_size_mapping[opt.datasets]

    if opt.action == "outlier_visualization":
        test_transform = transforms.Compose([transforms.ToTensor(),])
    else:
        if opt.datasets == "mnist":
            test_transform = transforms.Compose([transforms.ToTensor(),])
        elif opt.datasets == "FUB":
            test_transform = transforms.Compose([transforms.ToTensor(), transforms.CenterCrop((224, 288)),
                                                 transforms.CenterCrop((size, size)), ])
        elif opt.datasets in ["imagenet100", "imagenet100_small", "ImageNet100_Folder", "imagenet100_m",
                              "cub", "cars", "aircraft", "mit", "dtd", "flower", "imagenet1k"]:
            test_transform = transforms.Compose([transforms.ToTensor(), transforms.Resize((224, 224)),
                                                 normalize])
        else:
            test_transform = transforms.Compose([transforms.ToTensor(), normalize])

    data_fun = data_function_mapping_outliers[opt.datasets]
    label_dict = label_to_dict(osr_splits_outliers[opt.datasets][opt.trail], outliers=True)
    if class_idx is not None:
        classes = [osr_splits_outliers[opt.datasets][opt.trail][class_idx]]
    else:
        classes = osr_splits_outliers[opt.datasets][opt.trail]
    print(classes)
    if opt.datasets == "svhn":
        train = "test"
    else:
        train = False
    outlier_dataset = data_fun(root=data_root, train=train,
                               classes=classes, download=True, 
                               transform=test_transform, label_dict=label_dict)
    print("dataset size", len(outlier_dataset))
    return outlier_dataset


def sortFeatures(mixedFeatures, labels, num_classes):
        
    sortedFeatures = []
    for i in range(num_classes):
        sortedFeatures.append([])

    print("mixedFeatures", np.array(mixedFeatures).shape)
    
    for i, l in enumerate(labels):
        l = l.item()                         
        feature = mixedFeatures[i]
        feature = feature.reshape([-1])
        sortedFeatures[l].append(feature)
        
    # Attention the #samples for each class are different
    return sortedFeatures


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


def sample_distance_center_mahalanobis(train_data, train_labels, last_features, last_feature_labels, last_model, 
                                       label_dict, num_classes, subsample_transform, ratio_in, ratio_out, dataset_name, idx):

    last_features = np.squeeze(np.array(last_features))
    sorted_last_features = sortFeatures(last_features, last_feature_labels, num_classes)
    stats = feature_stats(sorted_last_features)
    cut_thresholds = mahalanobis_stats(sorted_last_features, stats, ratio_in, ratio_out)
    print("cut_thresholds", cut_thresholds)

    #train_data_sampled = copy.deepcopy(train_data)
    #train_labels_sampled = copy.deepcopy(train_labels)

    kept_indices = []
    features = [[] for i in range(num_classes)]

    for i, (sample, label)  in enumerate(zip(train_data, train_labels)):
        
        target = label_dict[str(label)]

        if "cifar10" in dataset_name or "cifar100" in dataset_name:
            sample= Image.fromarray(sample)
        elif "minist" in dataset_name:
            sample = Image.fromarray(sample, mode='L')
        elif "tinyimgnet" in dataset_name:
            sample = Image.fromarray(np.uint8(255 * sample))
        elif "svhn" in dataset_name:
            sample = Image.fromarray(np.transpose(sample, (1, 2, 0)))
        else:
            sample= Image.fromarray(sample)

        sample = subsample_transform(sample)
        sample = torch.unsqueeze(sample, dim=0)
        sample = sample.cuda()
        feature = last_model(sample)[idx]                # list from early to late layers
        feature = feature.cpu().detach().numpy()
        mu, var = stats[target]
        feature = np.squeeze(feature)
        m_dis = mahalanobis(feature, mu, np.linalg.inv(var))
        #print("feature, mu", np.mean(feature), np.mean(mu), np.mean(feature-mu), np.matmul((feature-mu), (feature-mu).T), m_dis)
        #print("m_distances", m_dis)
        cut_threshold1, cut_threshold2 = cut_thresholds[target]
        if m_dis > cut_threshold1 or m_dis < cut_threshold2:      # keep the most outside and most inside ones
            kept_indices.append(i)
        elif i % 2 == 0:
            kept_indices.append(i)

    return  kept_indices



def upsample_distance_center_mahalanobis(train_data, train_labels, last_features, last_feature_labels, last_model, 
                                         label_dict, num_classes, subsample_transform, portion_out, dataset_name, upsample_times, idx):

    last_features = np.squeeze(np.array(last_features))
    sorted_last_features = sortFeatures(last_features, last_feature_labels, num_classes)
    stats = feature_stats(sorted_last_features)
    cut_thresholds = mahalanobis_stats(sorted_last_features, stats, ratio_in=0, ratio_out=portion_out)
    print("cut_thresholds", cut_thresholds)

    #train_data_sampled = copy.deepcopy(train_data)
    #train_labels_sampled = copy.deepcopy(train_labels)

    upsample_indices = []
    features = [[] for i in range(num_classes)]

    for i, (sample, label)  in enumerate(zip(train_data, train_labels)):
        
        target = label_dict[str(label)]

        if "cifar10" in dataset_name or "cifar100" in dataset_name:
            sample= Image.fromarray(sample)
        elif "minist" in dataset_name:
            sample = Image.fromarray(sample, mode='L')
        elif "tinyimgnet" in dataset_name:
            sample = Image.fromarray(np.uint8(255 * sample))
        elif "svhn" in dataset_name:
            sample = Image.fromarray(np.transpose(sample, (1, 2, 0)))
        else:
            sample= Image.fromarray(sample)

        sample = subsample_transform(sample)
        sample = torch.unsqueeze(sample, dim=0)
        sample = sample.cuda()
        feature = last_model(sample)[idx]  
        feature = feature.cpu().detach().numpy()
        mu, var = stats[target]
        feature = np.squeeze(feature)
        m_dis = mahalanobis(feature, mu, np.linalg.inv(var))
        #print("feature, mu", np.mean(feature), np.mean(mu), np.mean(feature-mu), np.matmul((feature-mu), (feature-mu).T), m_dis)
        #print("m_distances", m_dis)
        cut_threshold1, _ = cut_thresholds[target]
        if m_dis > cut_threshold1:
            upsample_indices.append(i)

    kept_indices = upsample_indices * upsample_times + list(range(len(train_data)))

    return kept_indices



def normalFeatureReading(data_loader, model, opt):
    
    outputs_backbone = []
    outputs = []
    outputs_linear = []
    labels = []

    for i, (img, label) in enumerate(data_loader):
        
        print(i)
        if i > 5000:
            break

        if opt.method == "SupCon":
            output, output_encoder = model(img), model.encoder(img)
        else:
            output = model.encoder(img)

        outputs.append(output.detach().numpy())
        outputs_backbone.append(output_encoder.detach().numpy())

        labels.append(label.numpy())
    
    return output, output_encoder, labels


def featureMerge(featureList):
    
    featureMaps = []
    featureMaps_backbone = []
    labels = []

    for features in featureList:
        
        features, feature_backbone, labels_part = features
  
        featureMaps_backbone = featureMaps_backbone + feature_backbone
        featureMaps = featureMaps + features
        labels = labels + labels_part
        
    featureMaps_backbone = np.array(featureMaps_backbone, dtype=object)
    featureMaps = np.array(featureMaps, dtype=object)
    featureMaps_backbone = np.squeeze(featureMaps_backbone)
    featureMaps = np.squeeze(featureMaps)
    labels = np.squeeze(np.array(labels))

    return featureMaps, featureMaps_backbone, labels
    

def linear_classifier(mixedFeatures_train, labels_train, mixedFeatures_test, labels_test):

    clf = make_pipeline(StandardScaler(), SGDClassifier(max_iter=1000, tol=1e-3))
    clf.fit(mixedFeatures_train, labels_train)

    test_pred = clf.predict(mixedFeatures_test)
    print("Linear Classifier Accuracy: ", accuracy_plain(test_pred, labels_test))


def mixup_hybrid(batch_data1, batch_data2, labels1, labels2, alpha):

    lam = np.random.binomial(1, alpha)   #np.random.beta(alpha, alpha)
    new_imgs1 = []
    new_imgs2 = []
    new_labels = []

    # the batch_data is after two_transformation
    batch_data11 = batch_data1[0]
    batch_data12 = batch_data1[1]
    batch_data21 = batch_data2[0]
    batch_data22 = batch_data2[1]
    bz = len(batch_data11)

    for i, (img11, label11) in enumerate(zip(batch_data11, labels1)):
        idx = random.randint(0, bz-1)
        while labels2[idx] == label11:
            idx = random.randint(0, bz-1)
        img12 = batch_data12[idx]
        new_img = 0.5 * img11 + (lam) * 0.5 * img12 + (1-lam) * 0.5 * img11
        new_imgs1.append(new_img)

        img21 = batch_data21[i]
        img22 = batch_data22[idx]
        new_img = lam * img21 + (1 - lam) * img22
        new_imgs2.append(new_img)
        new_labels.append([label11.item(), labels2[idx].item()])

    new_imgs1 = torch.stack(new_imgs1)
    new_imgs2 = torch.stack(new_imgs2)
    mask = process_double_label(new_labels)
    
    return [new_imgs1, new_imgs2]
        

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


def mixup_positive_features(batch_features1, batch_features2, labels1, labels2, alpha, p, num_classes, method):

    bsz = batch_features1.shape[0]
    batch_features1_new = torch.ones_like(batch_features1)
    batch_features2_new = torch.ones_like(batch_features2)
    if method == "self_contract":
        lam = np.random.beta(alpha, alpha) + 1
        batch_features1_new = batch_features1 - p * batch_features2    #lam * batch_features1 + (1-lam) * batch_features2      
        batch_features2_new = batch_features2 - p * batch_features1   #lam * batch_features2 + (1-lam) * batch_features1      

        return batch_features1_new, batch_features2_new

    for c in range(num_classes):
        ci1 = torch.where(labels1 == c)
        ci2 = torch.where(labels2 == c)
        num_sample = len(ci1[0])

        c_features1 = batch_features1[ci1]
        c_features2 = batch_features2[ci2]
        c_features = torch.cat([c_features1, c_features2], dim=0)

        similarities = torch.matmul(c_features, c_features.T)
        similarities = similarities.cpu().detach().numpy()
        if method == "prob_similarity":
            probabilities = similarity_to_probability(similarities)
            idxs = selection_with_probability(probabilities)
        elif method == "min_similarity":
            idxs = selection_with_similarity(similarities, p=p)
        elif method == "random":
            idxs = selection_random(num_sample=num_sample)

        c_features_new = []
        lams = []
        for i, (feature, idx) in enumerate(zip(c_features, idxs)):
            if i < bsz:
                lam = np.random.beta(alpha, alpha)   #+ 1
                lams.append(lam)
            else:
                lam = lams[i-bsz]
            feature_new = lam * feature + (1 - lam) * (c_features[idx])
            c_features_new.append(feature_new)
            
        c_features_new = torch.stack(c_features_new, axis=0)
        #print("c_features_new", c_features_new.shape)
        f1, f2 = torch.split(c_features_new, [num_sample, num_sample], dim=0)
        batch_features1_new[ci1] = f1
        batch_features2_new[ci2] = f2

        #batch_features_new = torch.stack([batch_features1_new, batch_features2_new], axis=0)

    return batch_features1_new, batch_features2_new


def similarity_to_probability(similarity_matrix):

    """
    Here set small similarity with larger probability
    """

    #print("similarity_matrix", similarity_matrix)
    similarity_logit = np.exp(-similarity_matrix)
    np.fill_diagonal(similarity_logit, 0)
    probability_matrix = similarity_logit / np.sum(similarity_logit, axis=1, keepdims=True)

    #print("probability_matrix_inverse", probability_matrix)

    return probability_matrix


def selection_with_probability(probability_matrix):

    probability_matrix = probability_matrix * 1e6
    probability_matrix = probability_matrix.astype(int)
    #print(probability_matrix)

    idxs = []
    for i, p in enumerate(probability_matrix):
        #p = p[p!=0]
        idx = select_range(p, i)
        #print("p", p, "idx", idx)
        idxs.append(idx)

    return idxs


def select_range(prob_thresholds, f_idx):

    s = np.sum(prob_thresholds)
    a = random.randrange(0, s)
    prob_thresholds = [sum(prob_thresholds[:i]) for i in range(len(prob_thresholds)+1)][1:]
    prob_thresholds = np.array(prob_thresholds)
    r = np.where(a >= prob_thresholds)
    if len(r[0]) > 0:
        idx = r[0][-1] + 1
    else:
        idx = 0

    if idx == f_idx:
        idx = idx - 1
    
    return idx

    
def binary_random(p):

     """
     the probability of return 1 is p
     """
     return np.random.binomial(1, p)

    
def selection_with_similarity(similarity_matrix, p):

    idxs = []
    for i, ss in enumerate(similarity_matrix):
        
        if binary_random(p):
            idx = np.argmin(ss)
        else:
            idx = i
        
        idxs.append(idx)
    
    return idxs


def selection_random(num_sample):

    idxs = []
    for i in range(num_sample):
        idxs.append(random.randrange(0, num_sample))

    return idxs


def mixup_negative_features(batch_features1, batch_features2, labels1, labels2, alpha, method="max_similarity"):

    lam = np.random.beta(alpha, alpha)    # lam in [0, 1]
    if lam < 0.9:
        lam = np.random.beta(alpha, alpha)
    new_features1 = []
    new_features2 = []
    new_labels = []

    bz = len(batch_features1)
    similarities = torch.matmul(batch_features1, batch_features2.T)

    for i, (features11, label1) in enumerate(zip(batch_features1, labels1)):

        if method == "max_similarity":            # min_similarity
            features1_similarities = similarities[i].detach().cpu().numpy()
            sorted_indexes = np.argsort(features1_similarities)
            idx = sorted_indexes[-1]
            while labels2[idx] == label1:
                sorted_indexes = np.delete(sorted_indexes, -1)
                idx = sorted_indexes[-1]
        elif method == "random":
            idx = random.randint(0, bz-1)
            while labels2[idx] == label1:
            #print("idx", idx, len(labels2), labels2[idx], label11)
                idx = random.randint(0, bz-1)

        features12 = batch_features2[idx]
        new_feature = lam * features11 + (1 - lam) * features12
        new_features1.append(new_feature)

        features21 = batch_features2[i]
        features22 = batch_features1[idx]
        new_feature = lam * features21 + (1 - lam) * features22
        new_features2.append(new_feature)
        new_labels.append([label1.item(), labels2[idx].item()])

    new_features1 = torch.stack(new_features1)
    new_features2 = torch.stack(new_features2)

    return new_features1, new_features2


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


def backward_hook(module, grad_input, grad_output):
    
    global gradients
    #print("Backward Hook Running")
    gradients = grad_output
    #print (f"Gradient Size: {gradients[0].size()}")
    
    
def forward_hook(module, args, output):
    
    global activations
    #print("Forward Hook Running")
    activations = output
    #print (f"Gradient Size: {activations.size()}")


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


def sort_hooks_single(activations):

    activations_np = []

    for k in activations.keys():   
        activations_np.append(activations[k].cpu().detach())

    activations_np = torch.stack(activations_np)
    activations_np = activations_np.flatten(0, 1)        

    return activations_np


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
    

def iou(bbox1, bbox2):

    x1_min, y1_min, x1_max, y1_max = bbox1
    x2_min, y2_min, x2_max, y2_max = bbox2

    ix_min = max(x1_min, x2_min)
    iy_min = max(y1_min, y2_min)
    ix_max = min(x1_max, x2_max)
    iy_max = min(y1_max, y2_max)
    iw = max(0.0, ix_max - ix_min)
    ih = max(0.0, iy_max - iy_min)
    inter = iw * ih

    a1 = max(0.0, x1_max - x1_min) * max(0.0, y1_max - y1_min)
    a2 = max(0.0, x2_max - x2_min) * max(0.0, y2_max - y2_min)

    union = a1 + a2 - inter
    return float(inter / (union + 1e-20))


"""
def salient_maps(batch_data1, batch_data2, model):

    bsz, h, w = batch_data1.shape[0], batch_data1.shape[2], batch_data1.shape[3]
    # global variables
    #global gradients
    global activations
    
    # get the saliency map of the img
    criterion = SupConLoss(temperature=0.05)
    #backward_h = model.encoder.layer4[-1].register_full_backward_hook(backward_hook)   # 4, the last conv layer
    forward_h =  model.encoder.module.layer4[-1].register_forward_hook(forward_hook)          # 4    

    batch_data = torch.cat([batch_data1, batch_data2], dim=0)
    features = model(batch_data)
    features1, features2 = torch.split(features, [bsz, bsz], dim=0)
    features = torch.cat([features1.unsqueeze(1), features2.unsqueeze(1)], dim=1)
    loss = criterion(features)

    gradients = torch.autograd.grad(loss, activations)[0]    # what is the size of gradients if using parallel? TODO
    print("activation, ", activations.shape)
    print("gradients, ", gradients.shape)

    #loss.backward()
    #gradients = gradients[0]
    
    # average in each channel
    gradients = gradients.detach().cpu()
    activations = activations.detach().cpu()
    pooled_gradients = torch.mean(gradients, dim=[2,3])

    for i in range(activations.size()[1]):
        activations[:, i, :, :] *= pooled_gradients[:, i, None, None]
    
    activations = torch.mean(activations, dim=1)
    activations = F.relu(activations)
    activations = activations.detach()                     # the size of activations should be (bz, w, h)

    #del forward_hook
    #del backward_hook
    

    # search for the most activated areas
    #print("activations", gradients.shape)
    return activations


def layer_salient_maps(batch_data1, batch_data2, model):

    bsz, h, w = batch_data1.shape[0], batch_data1.shape[2], batch_data1.shape[3]
    global activations
    
    # get the saliency map of the img
    criterion = SupConLoss(temperature=0.05)
    forward_h =  model.encoder.layer3[-1].register_forward_hook(forward_hook)           

    batch_data = torch.cat([batch_data1, batch_data2], dim=0)
    features = model(batch_data)
    features1, features2 = torch.split(features, [bsz, bsz], dim=0)
    features = torch.cat([features1.unsqueeze(1), features2.unsqueeze(1)], dim=1)
    loss = criterion(features)

    gradients = torch.autograd.grad(loss, activations)[0] 
    
    # average in each channel
    gradients = gradients.detach().cpu()
    activations = activations.detach().cpu()
    
    with torch.no_grad():
        activation_maps = activations * F.relu(gradients)
        cam = torch.sum(activation_maps, dim=1).unsqueeze(0)
        cam = F.interpolate(cam, size=(h, w), mode="bilinear", align_corners=False)
        cam_min, cam_max = cam.min(), cam.max()
        norm_cam = (cam - cam_min).div(cam_max - cam_min + 1e-8).data
        norm_cam = torch.squeeze(norm_cam)

    return norm_cam

"""


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser('argument for training')
    parser.add_argument('--dataset', type=str, default=10)
    parser.add_argument('--datasets', type=str, default='aircraft',
                        choices=["cifar-10-100-10", "cifar-10-100-50", 'cifar10', "cifar100", "tinyimgnet",
                                 "imagenet100", "imagenet100_m", 'mnist', "svhn", "cub", "aircraft", "FUB"],
                        help='dataset')
    parser.add_argument("--augmix", type=bool, default=False)
    parser.add_argument("--action", type=str, default="training_supcon",
                        choices=["training_supcon", "trainging_linear", "testing_known", "testing_unknown",
                                 "feature_reading"])
    opt = parser.parse_args()

    dataset = get_train_datasets(opt)