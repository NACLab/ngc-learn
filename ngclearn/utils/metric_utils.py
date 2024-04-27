import jax
from jax import numpy as jnp, grad, jit, vmap, random, lax, nn
import os, sys
from functools import partial

@jit
def measure_sparsity(codes, tolerance=0.):
    """
    Calculates the sparsity (ratio) of an input matrix, assuming each row within
    it is a non-negative vector.

    Args:
        codes: matrix (shape: N x D) of non-negative codes to measure
            sparsity of (per row)

        tolerance: lowest number to consider as "empty"/non-existent (Default: 0.)

    Returns:
        sparsity measurements per code (output shape: N x 1)
    """
    m = (codes > tolerance).astype(jnp.float32)
    rho = jnp.sum(m, axis=1, keepdims=True)/(codes.shape[1] * 1.)
    return rho

@jit
def measure_ACC(mu, y): ## measures/calculates accuracy
    """
    Calculates the accuracy (ACC) given a matrix of predictions and matrix of targets.

    Args:
        mu: prediction (design) matrix

        y: target / ground-truth (design) matrix

    Returns:
        scalar accuracy score
    """
    guess = jnp.argmax(mu, axis=1)
    lab = jnp.argmax(y, axis=1)
    acc = jnp.sum( jnp.equal(guess, lab) )/(y.shape[0] * 1.)
    return acc

def measure_KLD(p_xHat, p_x, preserve_batch=False):
    """
    Measures the (raw) Kullback-Leibler divergence (KLD), assuming that the two
    input arguments contain valid probability distributions (in each row, if
    they are matrices). Note: If batch is preserved, this returns a column
    vector where each row is the KLD(x_pred, x_true) for that row's datapoint.

    | Formula:
    | KLD(p_xHat, p_x) = (1/N) [ sum_i(p_x * jnp.log(p_x)) - sum_i(p_x * jnp.log(p_xHat)) ]
    | where sum_i implies summing across dimensions of vector-space of p_x

    Args:
        p_xHat: predicted probabilities; (N x C matrix, where C is number of categories)

        p_x: ground true probabilities; (N x C matrix, where C is number of categories)

        preserve_batch: if True, will return one score per sample in batch
            (Default: False), otherwise, returns scalar mean score

    Returns:
        an (N x 1) column vector (if preserve_batch=True) OR (1,1) scalar otherwise
    """
    ## numerical control step
    offset = 1e-6
    _p_x = jnp.clip(p_x, offset, 1. - offset)
    _p_xHat = jnp.clip(p_xHat, offset, 1. - offset)
    ## calc raw KLD scores
    N = p_x.shape[1]
    term1 = jnp.sum(_p_x * jnp.log(_p_x), axis=1, keepdims=True) # * (1/N)
    term2 = -jnp.sum(_p_x * jnp.log(_p_xHat), axis=1, keepdims=True) # * (1/N)
    kld = (term1 + term2) * (1/N)
    if preserve_batch == False:
        kld = jnp.mean(kld)
    return kld

@partial(jit, static_argnums=[3])
def measure_CatNLL(p, x, offset=1e-7, preserve_batch=False):
    """
    Measures the negative Categorical log likelihood (Cat.NLL).  Note: If batch is
    preserved, this returns a column vector where each row is the
    Cat.NLL(p, x) for that row's datapoint.

    Args:
        p: predicted probabilities; (N x C matrix, where C is number of categories)

        x: true one-hot encoded targets; (N x C matrix, where C is number of categories)

        offset: factor to control for numerical stability (Default: 1e-7)

        preserve_batch: if True, will return one score per sample in batch
            (Default: False), otherwise, returns scalar mean score

    Returns:
        an (N x 1) column vector (if preserve_batch=True) OR (1,1) scalar otherwise
    """
    p_ = jnp.clip(p, offset, 1.0 - offset)
    loss = -(x * jnp.log(p_))
    nll = jnp.sum(loss, axis=1, keepdims=True) #/(y_true.shape[0] * 1.0)
    if preserve_batch == False:
        nll = jnp.mean(nll)
    return nll #tf.reduce_mean(nll)

@jit
def measure_MSE(mu, x, preserve_batch=False):
    """
    Measures mean squared error (MSE), or the negative Gaussian log likelihood
    with variance of 1.0. Note: If batch is preserved, this returns a column
    vector where each row is the MSE(mu, x) for that row's datapoint.

    Args:
        mu: predicted values (mean); (N x D matrix)

        x: target values (data); (N x D matrix)

        preserve_batch: if True, will return one score per sample in batch
            (Default: False), otherwise, returns scalar mean score

    Returns:
        an (N x 1) column vector (if preserve_batch=True) OR (1,1) scalar otherwise
    """
    diff = mu - x
    se = jnp.square(diff) ## squared error
    mse = jnp.sum(se, axis=1, keepdims=True) # technically se at this point
    if preserve_batch == False:
        mse = jnp.mean(mse) # this is proper mse
    return mse

@jit
def measure_BCE(p, x, offset=1e-7, preserve_batch=False): #1e-10
    """
    Calculates the negative Bernoulli log likelihood or binary cross entropy (BCE).
    Note: If batch is preserved, this returns a column vector where each row is
    the BCE(p, x) for that row's datapoint.

    Args:
        p: predicted probabilities of shape; (N x D matrix)

        x: target binary values (data) of shape; (N x D matrix)

        offset: factor to control for numerical stability (Default: 1e-7)

        preserve_batch: if True, will return one score per sample in batch
            (Default: False), otherwise, returns scalar mean score

    Returns:
        an (N x 1) column vector (if preserve_batch=True) OR (1,1) scalar otherwise
    """
    p_ = jnp.clip(p, offset, 1 - offset)
    bce = -jnp.sum(x * jnp.log(p_) + (1.0 - x) * jnp.log(1.0 - p_),axis=1, keepdims=True)
    if preserve_batch == False:
        bce = jnp.mean(bce)
    return bce
