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

def create_function(fun_name):
    fx = None
    dfx = None
    if fun_name == "tanh":
        fx = tanh
        dfx = d_tanh
    elif fun_name == "relu":
        fx = relu
        dfx = d_relu
    elif fun_name == "lrelu":
        fx = lrelu
        dfx = d_lrelu
    elif fun_name == "identity":
        fx = identity
        dfx = d_identity
    else:
        raise RuntimeError(
            "Activition function (" + fun_name + ") is not recognized/supported!"
            )
    return fx, dfx


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

@jit
def identity(x):
    return x + 0

@jit
def d_identity(x):
    return x * 0 + 1.

@jit
def relu(x):
    return nn.relu(x)

@jit
def d_relu(x):
    return (x >= 0.).astype(jnp.float32)

@jit
def tanh(x):
    return nn.tanh(x)

@jit
def d_tanh(x):
    tanh_x = nn.tanh(x)
    return -(tanh_x * tanh_x) + 1.0

@jit
def lrelu(x): ## activation fx
    return nn.leaky_relu(x)

@jit
def d_lrelu(x): ## deriv of fx (dampening function)
    m = (x >= 0.).astype(jnp.float32)
    dx = m + (1. - m) * 0.01
    return dx
