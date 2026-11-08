from __future__ import print_function

import math
import numpy as np
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from torchvision import transforms
from sklearn.metrics import roc_curve, auc


class TwoCropTransform:
    """Create two crops of the same image"""
    def __init__(self, transform, transform2=None, repeat=1, origin=False):
        self.transform = transform
        self.repeat = repeat
        self.origin = origin
        if transform2 is None:
            self.transform2 = transform
        else:
            self.transform2 = transform2

    def __call__(self, x):

        if self.repeat == 1:
            if self.origin:
                origin_transform = transforms.Compose([transforms.ToTensor(), transforms.CenterCrop(224)])
                return [self.transform(x), self.transform2(x), origin_transform(x)]
            return [self.transform(x), self.transform2(x)]
        else:
            transformed_x1 = []
            transformed_x2 = []
            for _ in range(self.repeat):
                transformed_x1.append(torch.unsqueeze(self.transform(x), dim=0))
                transformed_x2.append(torch.unsqueeze(self.transform2(x), dim=0))

            transformed_x1 = torch.cat(transformed_x1, dim=0)
            transformed_x2 = torch.cat(transformed_x2, dim=0)

            return [transformed_x1, transformed_x2]


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

"""
def accuracy(output, target):
    
    with torch.no_grad():
        
        batch_size = target.size(0)
        pred = torch.argmax(output, dim=1)
        score, _ = torch.max(output, dim=1)
        unequ = 0
        for i in range(batch_size):
            if pred[i] != target[i]:
                unequ += 1
        
        return 1 - unequ*1.0 / batch_size, pred, score
"""

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].float().sum()
            res.append(correct_k.mul_(100.0 / batch_size))
        return res
    


def accuracy_plain(predictions, targets):

    unequ = 0
    for p, t in zip(predictions, targets):
        
        if p != t:
            unequ += 1

    return 1 - unequ*1.0 / len(predictions)



def AUROC(labels, probs, opt):
    
    '''
    ROC: 
        X: False positive rate
        Y: True positive rate
    '''
    fpr, tpr, threholds = roc_curve(labels, probs)
    auroc = auc(fpr, tpr)

    # plot the AUROC curve
    #plt.plot([0, 1], [0, 1], linestyle="--")
    #plt.plot(fpr, tpr)
    #plt.ylabel("TPR (Sensitity)")
    #plt.xlabel("FPR (1 - Specificity)")

    #plt.savefig(opt.auroc_save_path)

    return auroc



def OSCR(x1, x2, pred, labels):

    """
    :param x1: open set score for each known class sample (B_k,)
    :param x2: open set score for each unknown class sample (B_u,)
    :param pred: predicted class for each known class sample (B_k,)
    :param labels: correct class for each known class sample (B_k,)
    :return: Open Set Classification Rate
    """

    x1, x2 = -x1, -x2

    # x1, x2 = np.max(pred_k, axis=1), np.max(pred_u, axis=1)
    # pred = np.argmax(pred_k, axis=1)

    correct = (pred == labels)
    m_x1 = np.zeros(len(x1))
    m_x1[pred == labels] = 1
    k_target = np.concatenate((m_x1, np.zeros(len(x2))), axis=0)
    u_target = np.concatenate((np.zeros(len(x1)), np.ones(len(x2))), axis=0)
    predict = np.concatenate((x1, x2), axis=0)
    n = len(predict)

    # Cutoffs are of prediction values

    CCR = [0 for x in range(n + 2)]
    FPR = [0 for x in range(n + 2)]

    idx = predict.argsort()

    s_k_target = k_target[idx]
    s_u_target = u_target[idx]

    for k in range(n - 1):
        CC = s_k_target[k + 1:].sum()
        FP = s_u_target[k:].sum()

        # True	Positive Rate
        CCR[k] = float(CC) / float(len(x1))
        # False Positive Rate
        FPR[k] = float(FP) / float(len(x2))

    CCR[n] = 0.0
    FPR[n] = 0.0
    CCR[n + 1] = 1.0
    FPR[n + 1] = 1.0

    # Positions of ROC curve (FPR, TPR)
    ROC = sorted(zip(FPR, CCR), reverse=True)

    OSCR = 0

    # Compute AUROC Using Trapezoidal Rule
    for j in range(n + 1):
        h = ROC[j][0] - ROC[j + 1][0]
        w = (ROC[j][1] + ROC[j + 1][1]) / 2.0

        OSCR = OSCR + h * w

    return OSCR



def adjust_learning_rate(args, optimizer, epoch):
    lr = args.learning_rate
    if args.cosine:
        eta_min = lr * (args.lr_decay_rate ** 3)
        lr = eta_min + (lr - eta_min) * (
                1 + math.cos(math.pi * epoch / args.epochs)) / 2
    else:
        steps = np.sum(epoch > np.asarray(args.lr_decay_epochs))
        if steps > 0:
            lr = lr * (args.lr_decay_rate ** steps)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def warmup_learning_rate(args, epoch, batch_id, total_batches, optimizer):
    if args.warm and epoch <= args.warm_epochs:
        p = (batch_id + (epoch - 1) * total_batches) / \
            (args.warm_epochs * total_batches)
        lr = args.warmup_from + p * (args.warmup_to - args.warmup_from)

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


def set_optimizer(opt, model):
    # optimizer = optim.SGD(model.parameters(),
    #                       lr=opt.learning_rate,
    #                       momentum=opt.momentum,
    #                       weight_decay=opt.weight_decay)
    optimizer = optim.Adam(model.parameters(), lr=opt.learning_rate)
    return optimizer


def save_model(model=None, linear=None, optimizer=None, opt=None, epoch=0, save_file=None):
    print('==> Saving...')
    if linear is not None:
        state = {
            'opt': opt,
            'model': model.state_dict(),
            "linear": linear.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,}
    else:
        state = {
            'opt': opt,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,}
    torch.save(state, save_file)
    del state


def plot_grad_flow(named_parameters, batchIdx, epoch):
    ave_grads = []
    layers = []
    for n, p in named_parameters:
        #print(n, p.grad)
        if (p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.cpu().abs().mean())
    
    fig = plt.figure()
    plt.plot(ave_grads, alpha=0.3, color='b')
    plt.hlines(0, 0, len(ave_grads)+1, linewidth=1, color="k" )
    plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(xmin=0, xmax=len(ave_grads))
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.savefig('./gradients/' + str(epoch) + "_" + str(batchIdx) + ".png")
    plt.close(fig)



def down_sampling(features, labels, ratio):
    
    length = len(features)

    indices = [i for i in range(length) if i % ratio == 0]
    features =  features[indices]
    labels = labels[indices]

    return features, labels



def feature_stats(inlier_features):

    stats = []
    for features in inlier_features:
        features = np.squeeze(np.array(features))
        mu = np.mean(features, axis=0)
        var = np.cov(features.astype(float), rowvar=False)

        stats.append((mu, var))
    
    return stats


def tau_step(opt, epoch):

    if opt.tau_set1 is None:
        return
        
    for i, e in enumerate(opt.tau_epochs):
        if epoch == e:
            opt.temp1 = opt.tau_set1[i]
            opt.temp2 = opt.tau_set2[i]
            print("Current temperature is ", opt.temp1, opt.temp2)


def tau_step_cosine(opt, epoch):

    opt.temp = (opt.temp_max - opt.temp_min) * (1 + math.cos(2 * math.pi * epoch / (opt.cosine_period))) / 2 + opt.temp_min
    print("Current temperature is ", opt.temp)


def tau_step_linear(opt, epoch):

    opt.temp = (opt.temp_max - opt.temp_min) / opt.epochs * (opt.epochs - epoch) + opt.temp_min
    print("Current temperature is ", opt.temp)


def tau_step_exp(opt, epoch):

    d = np.log(opt.temp_min / opt.temp_max) / opt.epochs 
    opt.temp = opt.temp_max * np.exp(epoch*d)
    print("Current temperature is ", opt.temp)


def CKA(features1, features2):
    
    xy = np.linalg.norm(np.matmul(features1.T, features2), "fro")
    xx = np.linalg.norm(np.matmul(features1.T, features1), "fro")
    yy = np.linalg.norm(np.matmul(features2.T, features2), "fro")

    xy = xy*xy

    print("xy", xy, "xx", xx, "yy", yy)

    return xy / xx / yy


def centering(K):
    n = K.shape[0]
    unit = np.ones([n, n])
    identity = np.eye(n)
    H = identity - unit / n
    return np.matmul(np.matmul(H, K), H)


def linear_HSIC(X, Y):
    L_X = np.matmul(X.T, X)
    L_Y = np.matmul(Y.T, Y)
    return np.sum(centering(L_X) * centering(L_Y))


def linear_CKA(X, Y):
    hsic = linear_HSIC(X, Y)
    var1 = np.sqrt(linear_HSIC(X, X))
    var2 = np.sqrt(linear_HSIC(Y, Y))

    print("hsic", hsic, "var1", var1, "var2", var2)

    return hsic / (var1 * var2)



def common_elements(list_a, list_b):

    common_list = [a for a in list_a if a in list_b]

    return common_list


def label_convert(labels, num_classes, coeff = 0.8, mode="smoothing"):

    converted_labels = []
    for i, label in enumerate(labels):
        converted_label = torch.zeros([num_classes])
        if mode == "smoothing":

            converted_label = converted_label + (1 - coeff) / (num_classes - 1)
            converted_label[label] = coeff

        elif mode == "one_hot":
            
            converted_label[label] = 1

        converted_labels.append(converted_label)

    converted_labels = torch.stack(converted_labels, dim=0)
    return converted_labels