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
        A data loader object, meant to allow sampling w/o replacement of one or
        more named design matrices.

        Args:
            design_matrices:  list of named data design matrices - [("name", matrix), ...]
            batch_size:  number of samples to place inside a mini-batch
            disable_shuffle:  if True, turns off sample shuffling (thus no sampling w/o replacement)

        @author Alexander Ororbia
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
            Yields a mini-batch of the form:  [("name", batch),("name",batch),...]
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
