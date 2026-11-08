import pickle
import os
import matplotlib.pyplot as plt


loss_files_path = "/home/zhi/projects/comprehensive_OSR_copy/loss_grad"
#loss_files_name = ["tinyimgnet_resnet18_vanilia__SimCLR_1.0_0.0_0.05_lam0_trail_0_128_256_center_loss",
#                   "tinyimgnet_resnet18_vanilia__SimCLR_1.0_1.0_0.05_lam0_trail_0_128_256_center_loss",
#                   "tinyimgnet_resnet18_mixup_positive_alpha_1.0_beta_1.0_layersaliencymix_2,3,4_SimCLR_1.0_1.0_0.05_lam0_trail_0_128_256_old_augmented_center_loss"]
loss_files_name = ["tinyimgnet_resnet18_vanilia__SimCLR_1.0_0.0_0.05_trail_5_128_256_loss",
                   "tinyimgnet_resnet18_vanilia__SimCLR_1.0_1.0_0.05_trail_5_128_256_old_augmented_loss",
                   "tinyimgnet_resnet18_mixup_positive_alpha_1.0_beta_1.0_layersaliencymix_2,3,4_SimCLR_1.0_1.0_0.05_trail_5_128_256_old_augmented_loss"]
#loss_files_name = ["tinyimgnet_resnet18_vanilia__SimCLR_1.0_0.0_0.05_trail_4_128_256_loss",
#                   "tinyimgnet_resnet18_vanilia__SimCLR_1.0_1.0_0.05_trail_4_128_256_old_augmented_loss",
#                   "tinyimgnet_resnet18_mixup_positive_alpha_1.0_beta_1.0_layersaliencymix_2,3,4_SimCLR_1.0_1.0_0.05_trail_4_128_256_old_augmented_loss"]
#loss_files_name = ["tinyimgnet_resnet18_vanilia__SimCLR_1.0_0.0_0.05_trail_1_128_256_loss",
#                   "tinyimgnet_resnet18_vanilia__SimCLR_1.0_1.0_0.05_trail_1_128_256_loss",
#                   "tinyimgnet_resnet18_mixup_positive_alpha_1.0_beta_1.0_layersaliencymix_2,3,4_SimCLR_1.0_1.0_0.05_trail_1_128_256_old_augmented_loss"]
#loss_files_name = ["cifar10_resnet18_vanilia__SimCLR_1.0_0.0_0.05_trail_5_128_256_old_augmented_loss",
#                   "cifar10_resnet18_vanilia__SimCLR_1.0_1.2_0.05_trail_5_128_256_old_augmented_loss",
#                   "cifar10_resnet18_mixup_positive_alpha_1.0_beta_1.0_layersaliencymix_3,4_SimCLR_1.0_1.2_0.05_trail_5_128_256_old_augmented_loss"]
#loss_files_name = ["cifar100_resnet18_vanilia__SimCLR_1.0_0.0_0.05_trail_0_128_256_old_augmented_loss",
#                   "cifar100_resnet18_vanilia__SimCLR_1.0_1.2_0.05_trail_0_128_256_old_augmented_loss",
#                   "cifar100_resnet18_mixup_positive_alpha_1.0_beta_1.0_layersaliencymix_2,3,4_SimCLR_1.0_1.2_0.05_trail_0_128_256_old_augmented_loss"]
#loss_files_name = ["imagenet100_m_resnet18_mixup_positive_alpha_1.0_beta_1.0_layersaliencymix_4_SimCLR_1.0_0_0.05_trail_0_128_230_old_augmented_center_loss",
#                   "imagenet100_m_resnet18_vanilia__SimCLR_1.0_1.0_0.05_trail_0_128_256_loss",
#                   "imagenet100_m_resnet18_mixup_positive_alpha_1.0_beta_1.0_layersaliencymix_4_SimCLR_1.0_0.5_0.05_trail_0_128_230_old_augmented_center_loss"]
loss_files_label = ["(sl)", "(sl+ssl)", "(GradMix)"]

colors = ["fuchsia", "yellowgreen", "r", "gold", "purple", "sienna"]
#["b", "g", "c", "lightgreen", "darkblue", "teal"]
styles = ['-', '-', '-']
markers = ['.', ',', '^']

epochs = list(range(600))

def differentiate(losses):

    length = len(losses)
    return [losses[i] - losses[i+1] for i in range(length) if i<length-1]


for i, (loss_file, label) in enumerate(zip(loss_files_name, loss_files_label)):
    print(loss_file)
    loss_file_path = os.path.join(loss_files_path, loss_file)
    with open(loss_file_path, "rb") as f:
        losses = pickle.load(f)

    if isinstance(losses, tuple):
        losses = losses[0]
    losses_all = []
    losses_sl = []
    losses_ssl = []
    for loss_epoch in losses:
        loss_all, loss_sl, loss_ssl = loss_epoch
        losses_all.append(loss_all)
        losses_sl.append(loss_sl)
        losses_ssl.append(loss_ssl)

    #plt.plot(losses_all, color=colors[0], linestyle=styles[i], label="losses_all")
    #print(sum(losses_sl)/len(losses_sl))
    plt.plot(epochs, losses_sl, color=colors[i*2], linewidth=2.0, label="losses_sl"+loss_files_label[i])
    plt.plot(epochs, losses_ssl, color=colors[i*2+1], linewidth=2.0, label="losses_ssl"+loss_files_label[i])
    d_sl = differentiate(losses_sl)
    d_ssl = differentiate(losses_ssl)
    #plt.plot(epochs[:-1], d_sl, color=colors[i * 2], linewidth=2.0, label="d_sl" + loss_files_label[i])
    #plt.plot(epochs[:-1], d_ssl, color=colors[i * 2 + 1], linewidth=2.0, label="d_ssl" + loss_files_label[i])

plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.title("Loss Dynamics ImageNet100")
plt.show()


"""
import argparse
import numpy as np

def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--grad_files_paths', type=str, default="./save/SupCon/tinyimgnet_models/tinyimgnet_resnet18_vanilia__SimCLR_1.0_1.0_0.05_lam0_trail_0_128_256_center")
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--metric", type=str, default="angle",
                        choices=["angle", "magnitude", "ratio"])
    opt = parser.parse_args()

    return opt


def sort_grads(grads_epoch):

    encoder_layer4 = []
    header1 = []
    header2 = []

    for grads in grads_epoch:
        el4, h1, h2 = grads
        encoder_layer4.append(el4.numpy())
        header1.append(h1.numpy())
        header2.append(h2.numpy())

    return encoder_layer4, header1, header2


def angle(grad_ssl, grad_sl):
    grad_ssl = grad_ssl.flatten()
    grad_sl = grad_sl.flatten()

    grad_total = grad_ssl + grad_sl

    return  np.dot(grad_total, grad_sl) / np.linalg.norm(grad_total) / np.linalg.norm(grad_sl)


def magnitude(grad_ssl, grad_sl):
    grad_ssl = grad_ssl.flatten()
    grad_sl = grad_sl.flatten()

    return 2 * np.linalg.norm(grad_ssl) * np.linalg.norm(grad_sl) / (np.linalg.norm(grad_ssl)*np.linalg.norm(grad_ssl) + np.linalg.norm(grad_sl)*np.linalg.norm(grad_sl))


def fraction(grad_ssl, grad_sl):
    grad_ssl = grad_ssl.flatten()
    grad_sl = grad_sl.flatten()

    return np.linalg.norm(grad_sl) / np.linalg.norm(grad_ssl)


if __name__ == "__main__":

    opt = parse_option()
    results = []

    for e in range(opt.num_epochs):
        grad_file_name = "grad_" + str(e)
        print(e)
        with open(os.path.join(opt.grad_files_paths, grad_file_name), "rb") as f:
            loss_ssl_grad, loss_sl_grad = pickle.load(f)

        encoder_layer4_ssl, header1_ssl, header2_ssl = sort_grads(loss_ssl_grad)
        encoder_layer4_sl, header1_sl, header2_sl = sort_grads(loss_sl_grad)

        if opt.metric == "angle":
            result = [angle(h2_ssl, h2_sl) for h2_ssl, h2_sl in zip(header2_ssl, header2_sl)]
        elif opt.metric == "magnitude":
            result  = [magnitude(h2_ssl, h2_sl) for h2_ssl, h2_sl in zip(header2_ssl, header2_sl)]
        elif opt.metric == "ratio":
            result = [fraction(h2_ssl, h2_sl) for h2_ssl, h2_sl in zip(header2_ssl, header2_sl)]
        #print("angles", angles)
        results.append(sum(result) / len(result))

    print("results", results)
"""