"""
Data functions and utilies.
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

def generate_patch_set(imgs, patch_shape, batch_size, center_patch=True):
    """
    Generates a set of patches from an array/list of image arrays (via
    random sampling with replacement).

    Args:
        imgs: the array of image arrays to sample from

        patch_shape: a 2-tuple of the form (pH = patch height, pW = patch width)

        batch_size: how many patches to extract/generate from source images

        center_patch: centers each patch by subtracting the patch mean (per-patch)

    Returns:
        an array (D x (pH * pW)), where each row is a flattened patch sample
    """
    pH, pW = patch_shape
    H, W, num_images = imgs.shape
    beginx = np.random.randint(0, W-pW, batch_size)
    beginy = np.random.randint(0, H-pW, batch_size)
    inputs_list = []
    # Get images randomly
    for i in range(batch_size):
        idx = np.random.randint(0, num_images)
        img = imgs[:, :, idx]
        clop = img[beginy[i]:beginy[i]+pW, beginx[i]:beginx[i]+pW].flatten()
        if center_patch == True:
            inputs_list.append(clop - np.mean(clop))
        else:
            inputs_list.append(clop)
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

            ensure_equal_batches: if True, ensures sampled batches are equal in size (Default = True).
                Note that this means the very last batch, if it's not the same size as the rest, will
                reuse random samples from previously seen batches (yielding a batch with a mix of
                vectors sampled with and without replacement).
    """
    def __init__(self, design_matrices, batch_size, disable_shuffle=False,
                 ensure_equal_batches=True):
        self.batch_size = batch_size
        self.ensure_equal_batches = ensure_equal_batches
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
        if self.disable_shuffle == False:
            self.ptrs = np.random.permutation(self.data_len)
        idx = 0
        while idx < len(self.ptrs): # go through each sample via the sampling pointer
            e_idx = idx + self.batch_size
            if e_idx > len(self.ptrs): # prevents reaching beyond length of dataset
                e_idx = len(self.ptrs)
            # extract sampling integer pointers
            indices = self.ptrs[idx:e_idx]
            if self.ensure_equal_batches == True:
                if indices.shape[0] < self.batch_size:
                    diff = self.batch_size - indices.shape[0]
                    indices = np.concatenate((indices, self.ptrs[0:diff]))
            # create the actual pattern vector batch block matrices
            data_batch = []
            for dname, dmatrix in self.design_matrices:
                x_batch = dmatrix[indices]
                data_batch.append((dname, x_batch))
            yield data_batch
            idx = e_idx

def binarized_shuffled_omniglot(out_dir): #, n_validation=1345):
    """
    Specialized function for the omniglot dataset.

    Note: this function has not been tested/fully integrated yet
    """
    def reshape_data(data):
        return data.reshape((-1, 28, 28)).reshape((-1, 28*28), order='fortran')
    omni_raw = scipy.io.loadmat(fname="{0}{1}".format(out_dir,'chardata.mat'))

    train_data = reshape_data(omni_raw['data'].T.astype('float32'))
    test_data = reshape_data(omni_raw['testdata'].T.astype('float32'))
    return train_data, test_data
