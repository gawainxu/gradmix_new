#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 14:38:44 2024

@author: zhi
"""

import pickle
import matplotlib.pyplot as plt


with open("loss_0", "rb") as f:
    loss = pickle.load(f)
    

loss0 = [l[0] for l in loss]
loss1 = [l[1] for l in loss]
loss2 = [l[2] for l in loss]

plt.plot(loss1)