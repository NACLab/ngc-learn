"""
General mathematical measurement/metric functions/utilities file.
"""
import tensorflow as tf
import numpy as np

def calc_firing_stats(frate_batch):
    """
    Calculate 1st and 2nd moment firing statistics from a batch of spike counts,
    i.e., mean firing rate and Fano factor.

    Args:
        frate_batch: a count batch matrix of shape (Q x D) where Q is number
            of trials and D is number of spiking neurons measured

    Returns:
        a (1 x D) mean firing rate vector and a (1 x D) Fano factor vector
    """
    # calc mean firing rate
    mu = tf.reduce_mean(frate_batch, axis=0, keepdims=True)
    # calc Fano factor
    dev = frate_batch - mu # deviation
    sig_sqr = tf.reduce_mean(dev * dev, axis=0, keepdims=True) # variance
    fano = sig_sqr / (mu + 1e-6)
    return mu, fano

def cat_nll(p, x, epsilon=0.0000001): #1e-7):
    """
    Measures the negative Categorical log likelihood

    Args:
        p: predicted probabilities

        x: true one-hot encoded targets

    Returns:
        an (N x 1) column vector, where each row is the Cat.NLL(x_pred, x_true)
        for that row's datapoint
    """
    p_ = tf.clip_by_value(p, epsilon, 1.0 - epsilon)
    loss = -(x * tf.math.log(p_))
    nll = tf.reduce_sum(loss,axis=1) #/(y_true.shape[0] * 1.0)
    return nll #tf.reduce_mean(nll)

def mse(mu, x):
    """
    Measures mean squared error (MSE), or the negative Gaussian log likelihood
    with variance of 1.0.

    Args:
        mu: predicted values (mean)

        x: target values (x/data)

    Returns:
        an (N x 1) column vector, where each row is the MSE(x_pred, x_true) for that row's datapoint
    """
    diff = mu - x
    se = tf.math.square(diff) #diff * diff # squared error
    # NLL = -( -se )
    return tf.reduce_sum(se, axis=1) # tf.math.reduce_mean(se)

def bce(p, x, offset=1e-7): #1e-10
    """
    Calculates the negative Bernoulli log likelihood or binary cross entropy (BCE).

    Args:
        p: predicted probabilities of shape (N x D)

        x: target binary values (data) of shape (N x D)

    Returns:
        an (N x 1) column vector, where each row is the BCE(p, x) for that row's datapoint
    """
    p_ = tf.clip_by_value(p, offset, 1 - offset)
    return -tf.reduce_sum(x * tf.math.log(p_) + (1.0 - x) * tf.math.log(1.0 - p_), axis=1)

def fast_log_loss(probs, y_ind):
    """
        Calculates negative Categorical log likelihood / cross entropy via a
        fast indexing approach (assumes targets/labels are integers or class
        indices for single-class one-hot encoding).

        Args:
            probs: predicted label probability distributions (one row per label)

            y_ind: label indices - can be either (D,) vector or (Dx1) column vector

        Returns:
            the scalar value of Cat.NLL(x_pred, x_true)
    """
    loss = 0.0
    y_ind_ = y_ind
    if len(y_ind_.shape) == 1:
        y_ind_ = tf.expand_dims(y_ind, 1)
    py = probs.numpy()
    for i in range(0, y_ind_.shape[0]):
        ti = y_ind_[i,0] # get ith target in sequence
        if ti >= 0: # entry for masked token, which should be non-negative
            py = probs[i,ti]
            if py <= 0.0:
                py = 1e-8
            loss += np.log(py) # all other columns in row i ( != ti) are 0, so do nothing
    return -loss # return negative summed log probs

def calc_ACC(T):
    """
    Calculates the average accuracy (ACC) given a task matrix T.

    Args:
        T: task matrix (containing accuracy values)

    Returns
        scalar ACC for T
    """
    acc = 0.0
    len_T = T.shape[0]
    for t in range(T.shape[1]):
        acc += T[len_T-1][t]
    return acc * (1.0 / (len_T * 1.0))

def calc_BWT(T):
    """
    Calculates the backward(s) transfer (BWT) given a task matrix T

    Args:
        T: task matrix (containing accuracy values)

    Returns
        scalar BWT for T
    """
    len_T = T.shape[0]
    bwt = 0.0
    #T_bot = T[len_T-1]
    for t in range(T.shape[1]-1):
        acc_tt = T[t,t]
        acc_fin = T[len_T-1,t]
        bwt_t = (acc_fin - acc_tt)
        bwt += bwt_t
    return bwt * (1.0 /((T.shape[1]-1) * 1.0))
