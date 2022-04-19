"""
Copyright (C) 2021 Alexander G. Ororbia II - All Rights Reserved
You may use, distribute and modify this code under the
terms of the BSD 3-clause license.

You should have received a copy of the BSD 3-clause license with
this file. If not, please write to: ago@cs.rit.edu
"""

import random
#import tensorflow as tf
import numpy as np
import io
import sys
import math

seed = 69
#tf.random.set_seed(seed=seed)
np.random.seed(seed)

class DataLoader(object):
    """
        :param design_matrices:  list of named data design matrices - [("name", matrix), ...]
        :param batch_size:  number of samples to place inside a mini-batch
        :disable_shuffle disable_shuffle:  if True turns off sample shuffling
        :return:  None
    """

    """
    Data Code
    self.xdata = np.loadtxt(self.x_fname, delimiter=delimiter)
    if flip_data is True:
        self.xdata = np.abs(self.xdata - 1.0)
    # OR
    self.xdata = np.load(x_fname)
    if flip_data is True:
        self.xdata = np.abs(self.xdata - 1.0)
    """

    def __init__(self, design_matrices, batch_size, disable_shuffle=False):
        self.batch_size = batch_size
        self.disable_shuffle = disable_shuffle
        self.design_matrices = design_matrices
        if len(design_matrices) < 1:
            print(" ERROR: design_matrices must contain at least one design matrix!")
            sys.exit(1)
        self.data_len = len( self.design_matrices[0][1] )
        self.ptrs = np.arange(0, self.data_len, 1)

    def __iter__(self):
        """
            yields a mini-batch of the form:  [("name", batch),("name",batch),...]
        """
        if self.disable_shuffle is False:
            self.ptrs = np.random.permutation(self.data_len)
        idx = 0
        while idx < len(self.ptrs): # go through each sample
            e_idx = idx + self.batch_size
            if e_idx > len(self.ptrs):
                e_idx = len(self.ptrs)
            indices = self.ptrs[idx:e_idx]
            data_batch = []
            for dname, dmatrix in self.design_matrices:
                x_batch = dmatrix[indices]
                data_batch.append((dname, x_batch))
            yield data_batch
            idx = e_idx
