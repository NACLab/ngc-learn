"""
Contains general mathematical measurement/metric functions.

@author Alexander Ororbia
"""
import tensorflow as tf
import numpy as np

def cat_nll(y_pred, y_true, epsilon=0.0000001): #1e-7):
    ''' Negative Categorical Log Likelihood '''
    y_pred_ = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
    loss = -(y_true * tf.math.log(y_pred_))
    nll = tf.reduce_sum(loss,axis=1,keepdims=True) #/(y_true.shape[0] * 1.0)
    return tf.reduce_mean(nll)

def mse(x_true, x_pred):
    ''' Mean Squared Error '''
    diff = x_pred - x_true
    se = diff * diff # squared error
    # NLL = -( -se )
    return tf.math.reduce_mean(se)

def bce(p, x, offset=1e-7): #1e-10
    """
        Calculates negative Bernoulli log likelihood or binary cross entropy (BCE)
    """
    p_ = tf.clip_by_value(p, offset, 1 - offset)
    return -tf.reduce_sum(x * tf.math.log(p_) + (1.0 - x) * tf.math.log(1.0 - p_), axis=1)

def fast_log_loss(probs, y_ind_):
    """
        Calculates negative Categorical log likelihood / cross entropy via a
        fast indexing approach (assumes targets/labels are integers or class
        indices for single-class one-hot encoding)
    """
    loss = 0.0
    y_ind = tf.expand_dims(y_ind_, 1)
    py = probs.numpy()
    for i in range(0, y_ind.shape[0]):
        ti = y_ind[i,0] # get ith target in sequence
        if ti >= 0: # entry for masked token, which should be non-negative
            py = probs[i,ti]
            if py <= 0.0:
                py = 1e-8
            loss += np.log(py) # all other columns in row i ( != ti) are 0, so do nothing
    return -loss # return negative summed log probs

def calc_ACC(T):
    ''' Calculates average accuracy given a task matrix T'''
    acc = 0.0
    len_T = T.shape[0]
    for t in range(T.shape[1]):
        acc += T[len_T-1][t]
    return acc * (1.0 / (len_T * 1.0))

def calc_BWT(T):
    ''' Calculates backward transfer given a task matrix T'''
    len_T = T.shape[0]
    bwt = 0.0
    #T_bot = T[len_T-1]
    for t in range(T.shape[1]-1):
        acc_tt = T[t,t]
        acc_fin = T[len_T-1,t]
        bwt_t = (acc_fin - acc_tt)
        bwt += bwt_t
    return bwt * (1.0 /((T.shape[1]-1) * 1.0))
