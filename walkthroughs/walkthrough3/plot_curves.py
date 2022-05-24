"""
Copyright (C) 2021 Alexander G. Ororbia II - All Rights Reserved
You may use, distribute and modify this code under the
terms of the BSD 3-clause license.

You should have received a copy of the BSD 3-clause license with
this file. If not, please write to: ago@cs.rit.edu
"""
import os
import sys
import numpy as np
import time
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
#import ngclearn.utils.io_utils as io_tools

"""
################################################################################
Demo/Tutorial #3 File:
Plots learning curves for an NGC model trained on the MNIST database.

Usage:
$ python plot_curves.py

@author Alexander Ororbia
################################################################################
"""

colors = ["red", "blue"]

# plot loss learning curves
y = 1.0 - np.load("gncn_t1_ffm/Acc0.npy")
vy = 1.0 - np.load("gncn_t1_ffm/vAcc0.npy")
y = y * 100.0
vy = vy * 100.0
x_iter = np.asarray(list(range(0, y.shape[0])))
fontSize = 20
plt.plot(x_iter, y, '-', color=colors[0])
plt.plot(x_iter, vy, '-', color=colors[1])
plt.xlabel("Epoch", fontsize=fontSize)
plt.ylabel("Classification Error", fontsize=fontSize)
plt.grid()
loss = mpatches.Patch(color=colors[0], label='Acc')
vloss = mpatches.Patch(color=colors[1], label='V-Acc')
plt.legend(handles=[loss, vloss], fontsize=13, ncol=2,borderaxespad=0, frameon=False,
           loc='upper center', bbox_to_anchor=(0.5, -0.175))#, prop=fontP)
plt.tight_layout()
plt.savefig("gncn_t1_ffm/mnist_learning_curves.jpg")
plt.clf()
