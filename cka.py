import math
import os
import argparse
import numpy as np
import torch
import pickle


def parse_option():

    parser = argparse.ArgumentParser('argument for feature comparision')

    parser.add_argument('--datasets', type=str, default='voc',
                        choices=["cifar100", 'cifar10', "tinyimgnet", 'mnist', "svhn", "voc"], help='dataset')
    parser.add_argument("--feature_path1", type=str, default="/features/cifar10_resnet18_trail_0_128_0.005_train")
    parser.add_argument("--feature_path2", type=str, default="/features/cifar10_resnet18_trail_0_128_0.01_train")
    parser.add_argument("--num_classes", type=int, default=6, help="Number of classes")
    
    opt = parser.parse_args()
    opt.main_dir = os.getcwd()
    opt.feature_path1 = opt.main_dir + opt.feature_path1
    opt.feature_path2 = opt.main_dir + opt.feature_path2
    
    return opt

def hsic(matrix_x: np.ndarray, matrix_y: np.ndarray) -> float:
    n = matrix_x.shape[0]
    matrix_h = np.identity(n) - (1.0 / n) * np.ones((n, n))

    x_times_h = np.matmul(matrix_x, matrix_h)
    y_times_h = np.matmul(matrix_y, matrix_h)

    return 1.0 / ((n - 1) ** 2) * np.trace(np.matmul(x_times_h, y_times_h))


def linear_cka(matrix_x: np.ndarray, matrix_y: np.ndarray) -> float:
    
    if matrix_x.ndim > 2:
        matrix_x = matrix_x.reshape([matrix_x.shape[0], -1])
    
    if matrix_y.ndim > 2:
        matrix_y = matrix_y.reshape([matrix_y.shape[0], -1])
    
    # First center the columns
    matrix_x = matrix_x - np.mean(matrix_x, 0)
    matrix_y = matrix_y - np.mean(matrix_y, 0)

    matrix_x = np.matmul(matrix_x, matrix_x.T)
    matrix_y = np.matmul(matrix_y, matrix_y.T)

    matrix_h = hsic(matrix_x=matrix_x, matrix_y=matrix_y)
    matrix_x = np.sqrt(hsic(matrix_x=matrix_x, matrix_y=matrix_x))
    matrix_y = np.sqrt(hsic(matrix_x=matrix_y, matrix_y=matrix_y))
    return matrix_h / (matrix_x * matrix_y)


class TorchCKA(object):
    def __init__(self, device):
        self.device = device

    def centering(self, K):
        n = K.shape[0]
        unit = torch.ones([n, n], device=self.device)
        identity = torch.eye(n, device=self.device)
        H = identity - unit / n
        return torch.matmul(torch.matmul(H, K), H)

    def rbf(self, X, sigma=None):
        GX = torch.matmul(X, X.T)
        KX = torch.diag(GX) - GX + (torch.diag(GX) - GX).T
        if sigma is None:
            mdist = torch.median(KX[KX != 0])
            sigma = math.sqrt(mdist)
        KX *= -0.5 / (sigma * sigma)
        KX = torch.exp(KX)
        return KX

    def kernel_HSIC(self, X, Y, sigma):
        return torch.sum(self.centering(self.rbf(X, sigma)) * self.centering(self.rbf(Y, sigma)))

    def linear_HSIC(self, X, Y):
        L_X = torch.matmul(X, X.T)
        L_Y = torch.matmul(Y, Y.T)
        return torch.sum(self.centering(L_X) * self.centering(L_Y))

    def linear_CKA(self, X, Y):
        hsic = self.linear_HSIC(X, Y)
        var1 = torch.sqrt(self.linear_HSIC(X, X))
        var2 = torch.sqrt(self.linear_HSIC(Y, Y))

        return hsic / (var1 * var2)

    def kernel_CKA(self, X, Y, sigma=None):
        hsic = self.kernel_HSIC(X, Y, sigma)
        var1 = torch.sqrt(self.kernel_HSIC(X, X, sigma))
        var2 = torch.sqrt(self.kernel_HSIC(Y, Y, sigma))
        return hsic / (var1 * var2)
    
    

def sort_features(opt, features, labels):
   
    sorted_features = []
    min_num_samples = 1000

    for i in range(opt.num_classes):
        features_i = []
        for f, l in zip(features, labels):
            if l == i:
                features_i.append(f)
        if len(features_i) < min_num_samples:
            min_num_samples = len(features_i)
        sorted_features.append(features_i)
        
    #sorted_features = [features[:min_num_samples] for features in sorted_features]
    sorted_features = np.array(sorted_features)
    return sorted_features


def linear_cka_gpt(X, Y, eps=1e-12):
    X = np.asarray(X)
    Y = np.asarray(Y)

    if X.ndim > 2:
        X = X.reshape(X.shape[0], -1)
    if Y.ndim > 2:
        Y = Y.reshape(Y.shape[0], -1)

    X = X - X.mean(axis=0, keepdims=True)
    Y = Y - Y.mean(axis=0, keepdims=True)

    hsic = np.linalg.norm(X.T @ Y, ord="fro") ** 2
    var_x = np.linalg.norm(X.T @ X, ord="fro")
    var_y = np.linalg.norm(Y.T @ Y, ord="fro")

    return hsic / (var_x * var_y + eps)


if __name__ == "__main__":
    
    opt = parse_option()
    
    with open(opt.feature_path1, "rb") as f:
        features1, _, labels1 = pickle.load(f)
        
    with open(opt.feature_path2, "rb") as f:
        features2, _, labels2 = pickle.load(f)


    sorted_features1 = sort_features(opt, features1, labels1)
    sorted_features2 = sort_features(opt, features2, labels2)
    cka = []
    for i in range(opt.num_classes):
        cka_i = linear_cka_gpt(sorted_features1[i], sorted_features2[i])
        print("class", i, "cka", cka_i)
        cka.append(cka_i)     
    print("mean cka", sum(cka)/len(cka))


    cka_gpt = linear_cka_gpt(features1[:6001], features2[:6001])
    print("cka gpt", cka_gpt)
    
    
    
    
