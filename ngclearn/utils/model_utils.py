import jax
from jax import numpy as jnp, grad, jit, vmap, random, lax
import os, sys
from functools import partial

@jit
def calc_acc(mu, y): ## calculates accuracy
    guess = jnp.argmax(mu, axis=1)
    lab = jnp.argmax(y, axis=1)
    acc = jnp.sum( jnp.equal(guess, lab) )/(y.shape[0] * 1.)
    return acc
    
def initialize_params(dkey, initKernel, shape):
    initType, *args = initKernel # get out arguments of initialization kernel
    params = None
    if initType == "hollow":
        eyeScale, _ = args
        dim = shape[1]
        params = (1. - jnp.eye(dim)) * eyeScale
    elif initType == "eye":
        eyeScale, _ = args
        dim = shape[1]
        params = jnp.eye(dim) * eyeScale
    else: # uniform
        lb, ub = args
        params = random.uniform(dkey, shape, minval=lb, maxval=ub)
    return params

@partial(jit, static_argnums=[2, 3])
def normalize_matrix(M, wnorm, ord=1, axis=0):
    '''
    Normalizes the synapses to have a particular norm across each vector span.

    Args:
        M: (2D) matrix to normalize

        wnorm: target norm for each

        ord: order of norm to use in normalization

        axis: 0 (apply to column vectors), 1 (apply to row vectors)
    '''
    wAbsSum = jnp.sum(jnp.abs(M), axis=axis, keepdims=True)
    m = (wAbsSum == 0.).astype(dtype=jnp.float32)
    wAbsSum = wAbsSum * (1. - m) + m
    #wAbsSum[wAbsSum == 0.] = 1.
    _M = M * (wnorm/wAbsSum)
    return _M


@jit
def one_hot(P):
    '''
    Converts a matrix of probabilities to a corresponding binary one-hot matrix
    (each row is a one-hot encoding).

    Args:
        P: a probability matrix where each row corresponds to a particular
            data probability vector
    '''
    nC = P.shape[1] # compute number of dimensions/classes
    p_t = jnp.argmax(P, axis=1)
    return nn.one_hot(p_t, num_classes=nC, dtype=jnp.float32)
