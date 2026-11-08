# visualize the feature ensemble results
#https://stackoverflow.com/questions/4700614/how-to-put-the-legend-outside-the-plot

import matplotlib.pyplot as plt
import numpy as np


"""
values_wo = {"accuracy": np.array([ 95.73,	96.2, 96.23, 96.78]),
          "auroc": np.array([86.62,	87.48, 87.8, 88.41,])} 


values_w = {"accuracy": np.array([ 96.62,	96.75,	97.1,	97.32]),
          "auroc": np.array([89.15,	89.41,	90.83,	91.57])} 

beta = ["0.2", "0.4", "0.6", "0.8"]

plt.figure(figsize=(5, 4), dpi=300)

# Plotting both curves
plt.plot(values_w["auroc"], values_w["accuracy"], label='w GradMix', color='blue', marker='o')
plt.plot(values_wo["auroc"],  values_wo["accuracy"], label='wo GradMix', color='red', marker='o')

plt.margins(x=0.12, y=0.15)


for i, (x, y) in enumerate(zip(values_w["auroc"], values_w["accuracy"])):
    plt.annotate(
        beta[i],           # The label text
        xy=(x, y),               # The point to annotate
        textcoords="offset points", # How to position the text
        xytext=(0, 10),          # Distance from text to point (x,y)
        va='top', ha='center',             # Horizontal alignment
        fontsize=12,
        color='k'
    )


for i, (x, y) in enumerate(zip(values_wo["auroc"], values_wo["accuracy"])):
    plt.annotate(
        beta[i],           # The label text
        xy=(x, y),               # The point to annotate
        textcoords="offset points", # How to position the text
        xytext=(0, 10),          # Distance from text to point (x,y)
        va='top', ha='center',             # Horizontal alignment
        fontsize=12,
        color='k'
    )


plt.xlabel('AUROC (%)', fontsize=12)
plt.ylabel('Accuracy (%)', fontsize=12)
#plt.title(r'AUROC vs. Accuracy with varying $\beta$')
plt.legend()

plt.savefig("./plots/preto_cifar.pdf", bbox_inches="tight")

"""


values_wo = {"accuracy": np.array([0.4938, 0.6872, 0.694, 0.7078]),
          "auroc": np.array([0.7078, 0.8332, 0.82, 0.834])} 


values_w = {"accuracy": np.array([0.5964, 0.741, 0.7548, 0.7674]),
          "auroc": np.array([0.7498, 0.8494, 0.8495, 0.8462])} 

beta = ["0.2", "0.4", "0.6", "0.8"]

plt.figure(figsize=(5, 4), dpi=300)

# Plotting both curves
plt.plot(values_w["auroc"], values_w["accuracy"], label='w GradMix', color='blue', marker='o')
plt.plot(values_wo["auroc"],  values_wo["accuracy"], label='wo GradMix', color='red', marker='o')

plt.margins(x=0.12, y=0.15)


for i, (x, y) in enumerate(zip(values_w["auroc"], values_w["accuracy"])):
    plt.annotate(
        beta[i],           # The label text
        xy=(x, y),               # The point to annotate
        textcoords="offset points", # How to position the text
        xytext=(0, 10),          # Distance from text to point (x,y)
        va='top', ha='center',             # Horizontal alignment
        fontsize=12,
        color='k'
    )


for i, (x, y) in enumerate(zip(values_wo["auroc"], values_wo["accuracy"])):
    plt.annotate(
        beta[i],           # The label text
        xy=(x, y),               # The point to annotate
        textcoords="offset points", # How to position the text
        xytext=(0, 10),          # Distance from text to point (x,y)
        va='top', ha='center',             # Horizontal alignment
        fontsize=12,
        color='k'
    )


plt.xlabel('AUROC (%)', fontsize=12)
plt.ylabel('Accuracy (%)', fontsize=12)
#plt.title(r'AUROC vs. Accuracy with varying $\beta$')
plt.legend()

plt.savefig("./plots/preto_imagenet.pdf", bbox_inches="tight")