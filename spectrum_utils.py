#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 31 17:21:21 2025

@author: zhi
"""

import numpy as np
import pickle
import matplotlib.pyplot as plt


if __name__ == "__main__":
    
    """
    # for inter
    data_paths = ["./features/1.0_inter", "./features/0.5_inter", "./features/0.1_inter", "./features/0.05_inter",
                  "./features/0.01_inter", "./features/0.005_inter"]
    ms = []
    num_files = len(data_paths)
    titles = [r'$\tau = 1.0$', r'$\tau = 0.5$', r'$\tau = 0.1$', 
              r'$\tau = 0.05$', r'$\tau = 0.01$', r'$\tau = 0.005$']
    
    for dp in data_paths:
        with open(dp, "rb") as f:
            m = pickle.load(f)
        ms.append(m)
        
    rows, cols = 1, num_files
    fig, axes = plt.subplots(rows, cols, figsize=(12, 8))
    axes = axes.flatten()

    vmin = min(m.min() for m in ms)
    vmax = max(m.max() for m in ms)
    cmap = 'viridis'

    for ax, matrix, title in zip(axes, ms, titles):
        im = ax.imshow(matrix, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(title)

    fig.tight_layout(rect=[0, 0.07, 1, 1])
    cbar = fig.colorbar(im, ax=axes, orientation='horizontal', fraction=0.02, pad=0.05, aspect=70)
    cbar.set_label('Value')

    plt.savefig("inter.pdf", bbox_inches='tight')
    plt.show()
    """
    
    # for intra
    data_paths = ["./features/1.0_intra", "./features/0.5_intra", "./features/0.1_intra", "./features/0.05_intra",
                  "./features/0.01_intra", "./features/0.005_intra"]
    
    ms = []
    for dp in data_paths:
        with open(dp, "rb") as f:
            mc = pickle.load(f)
        mc = np.array(mc)
        mc = np.mean(mc, axis=0)
        ms.append(mc)
    
    num_files = len(data_paths)
    titles = [r'$\tau = 1.0$', r'$\tau = 0.5$', r'$\tau = 0.1$', 
              r'$\tau = 0.05$', r'$\tau = 0.01$', r'$\tau = 0.005$']
        
    rows, cols = 1, num_files
    fig, axes = plt.subplots(rows, cols, figsize=(12, 8))
    axes = axes.flatten()

    vmin = min(m.min() for m in ms)
    vmax = max(m.max() for m in ms)
    cmap = 'viridis'

    for ax, matrix, title in zip(axes, ms, titles):
        im = ax.imshow(matrix, cmap=cmap, vmin=vmin, vmax=vmax)
        print(np.mean(matrix))
        ax.set_title(title)

    fig.tight_layout(rect=[0, 0.07, 1, 1])
    cbar = fig.colorbar(im, ax=axes, orientation='horizontal', fraction=0.02, pad=0.05, aspect=70)
    cbar.set_label('Value')

    plt.savefig("intra.pdf", bbox_inches='tight')
    plt.show()
    
    
    
    