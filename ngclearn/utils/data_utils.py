"""
Data functions and utilies.

@author Alexander Ororbia
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

def create_patches(imgs, pH, pW, batch_size):
    """
    Generates a set of patches from an array/list of image arrays (via
    random sampling with replacement).

    Args:
        imgs: the array of image arrays to sample from

        pH: patch height

        pW: patch width

        batch_size: how many patches to extract/generate from source images

    Returns:
        an array (D x (pH * pW)), where each row is a flattened patch sample
    """
    H, W, num_images = imgs.shape
    beginx = np.random.randint(0, W-sz, batch_size)
    beginy = np.random.randint(0, H-sz, batch_size)
    inputs_list = []
    # Get images randomly
    for i in range(batch_size):
        idx = np.random.randint(0, num_images)
        img = imgs[:, :, idx]
        clop = img[beginy[i]:beginy[i]+sz, beginx[i]:beginx[i]+sz].flatten()
        inputs_list.append(clop - np.mean(clop))
    patches = np.array(inputs_list, dtype=np.float32) # Input image patches
    return patches

class DataLoader(object):
    """
        A data loader object, meant to allow sampling w/o replacement of one or
        more named design matrices. Note that this object is iterable (and
        implements an __iter__() method).

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
