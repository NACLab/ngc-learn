"""
Copyright (C) 2021 Alexander G. Ororbia II - All Rights Reserved
You may use, distribute and modify this code under the
terms of the BSD 3-clause license.

You should have received a copy of the BSD 3-clause license with
this file. If not, please write to: ago@cs.rit.edu
"""

"""
Contains some specialized loaders for various dataset benchmarks

@author: Alex Ororbia
"""
import tensorflow as tf
import numpy as np
import scipy
#import tensorflow.contrib.eager as tfe
#tf.enable_eager_execution()
#tf.executing_eagerly()
seed = 69
#tf.random.set_random_seed(seed=seed)

tf.random.set_seed(seed=seed)
np.random.seed(seed)


def binarized_shuffled_omniglot(out_dir): #, n_validation=1345):
    def reshape_data(data):
        return data.reshape((-1, 28, 28)).reshape((-1, 28*28), order='fortran')
    omni_raw = scipy.io.loadmat(fname="{0}{1}".format(out_dir,'chardata.mat'))

    train_data = reshape_data(omni_raw['data'].T.astype('float32'))
    test_data = reshape_data(omni_raw['testdata'].T.astype('float32'))
    return train_data, test_data
