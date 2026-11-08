#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 13 15:49:22 2025

@author: zhi
"""

import pickle
import matplotlib.pyplot as plt
import numpy as np


def energy(energy_list):
    
    new_energy_list = [ area_in / (area_obj + area_out) for area_in, area_out, area_obj in energy_list]
    return np.array(new_energy_list)


with open("./features/energy_entropy_ssl_0.01_True", "rb") as f:
    bg_entropy_list_ssl, energy_list_ssl = pickle.load(f)

bg_entropy_list_ssl = np.array(bg_entropy_list_ssl)
energy_list_ssl = np.array(energy_list_ssl)

indices = np.where(bg_entropy_list_ssl < 1000)
bg_entropy_list_ssl = bg_entropy_list_ssl[indices]
energy_list_ssl = energy_list_ssl[indices]
energy_list_ssl = energy(energy_list_ssl)

indices = np.where(energy_list_ssl < 100)
bg_entropy_list_ssl = bg_entropy_list_ssl[indices]
energy_list_ssl = energy_list_ssl[indices]


with open("./features/energy_entropy_supcon_0.01_True", "rb") as f:
    bg_entropy_list_supcon, energy_list_supcon = pickle.load(f)

bg_entropy_list_supcon = np.array(bg_entropy_list_supcon)
energy_list_supcon = np.array(energy_list_supcon)

indices = np.where(bg_entropy_list_supcon < 1000)
bg_entropy_list_supcon = bg_entropy_list_supcon[indices]
energy_list_supcon = energy_list_supcon[indices]
energy_list_supcon = energy(energy_list_supcon)

indices = np.where(energy_list_supcon < 100)
bg_entropy_list_supcon = bg_entropy_list_supcon[indices]
energy_list_supcon = energy_list_supcon[indices]


with open("./features/energy_entropy_grad_0.01_True", "rb") as f:
    bg_entropy_list_grad, energy_list_grad = pickle.load(f)

bg_entropy_list_grad = np.array(bg_entropy_list_grad)
energy_list_grad = np.array(energy_list_grad)

indices = np.where(bg_entropy_list_grad < 1000)
bg_entropy_list_grad = bg_entropy_list_grad[indices]
energy_list_grad = energy_list_grad[indices]
energy_list_grad = energy(energy_list_grad)

indices = np.where(energy_list_grad < 100)
bg_entropy_list_grad = bg_entropy_list_grad[indices]
energy_list_grad = energy_list_grad[indices]



def means(bg_entropy_list, energy_list, range_x):
    
    ms = []
    step = range_x[1] - range_x[0]

    for i in range_x:
        
        indices = np.where((bg_entropy_list <= i) & (bg_entropy_list >= i-step))
        ms.append(np.mean(energy_list[indices]))
        
    return ms


range_x1 = np.arange(0.1, 1, 0.08) 
ms_ssl1 = means(bg_entropy_list_ssl, energy_list_ssl, range_x1)
ms_supcon1 = means(bg_entropy_list_supcon, energy_list_supcon, range_x1)
ms_grad1 = means(bg_entropy_list_grad, energy_list_grad,range_x1)

"""
indices = []
for i, _ in enumerate(range_x1):
    
    if ms_ssl1[i] == ms_ssl1[i]:
        indices.append(i)

range_x1 = [range_x1[i] for i in indices]
ms_ssl1 = [ms_ssl1[i] for i in indices]
ms_supcon1 = [ms_supcon1[i] for i in indices]
ms_grad1 = [ms_grad1[i] for i in indices]
        
"""

range_x2 = np.arange(1, 9, 0.5)
ms_ssl2 = means(bg_entropy_list_ssl, energy_list_ssl, range_x2)
ms_supcon2 = means(bg_entropy_list_supcon, energy_list_supcon, range_x2)
ms_grad2 = means(bg_entropy_list_grad, energy_list_grad,range_x2)

ms_ssl = ms_ssl1 + ms_ssl2
ms_supcon = ms_supcon1 + ms_supcon2
ms_grad = ms_grad1 + ms_grad2
range_x = np.concatenate([range_x1, range_x2], axis=0)

from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(range_x, ms_supcon , "-*", label="SupCon")
ax.plot(range_x, ms_ssl, "-*", label="+ SSL")
ax.plot(range_x, ms_grad, "-*", label="+ SSL & GradMix")
ax.set_xlabel("Background Entropy", fontsize=24)
ax.set_ylabel("IoU", fontsize=24)
ax.legend(fontsize=18)
ax.tick_params(axis='both', which='major', labelsize=14)

"""
# Create inset axes
axins = inset_axes(ax, width="40%", height="30%", loc="upper right")  # inset position
axins.plot(range_x, ms_ssl)
axins.set_xlim(range_x1[0], range_x1[-1])
axins.set_ylim(ms_ssl1[0], ms_ssl1[-1])
axins.set_xticks([])
axins.set_yticks([])
"""

plt.savefig("localization_entropy_001.pdf",  bbox_inches='tight')