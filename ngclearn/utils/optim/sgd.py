import sys, getopt, optparse
import numpy as np
from jax import jit, numpy as jnp, random, nn, lax
from functools import partial
import time

@jit
def _update(param, grad, lr):
    _param = param - lr * grad
    return _param

class SGD():
    """
    Implements stochastic gradient descent (SGD) as a decoupled update rule
    given adjustments produced by a credit assignment algorithm.

    -- Arguments --
    :param learning_rate: step size coefficient for SGD update

    @author: Alexander G. Ororbia II
    """
    def __init__(self, learning_rate=0.001):
        self.eta = learning_rate
        self.time = 0.

    def update(self, theta, updates): ## apply adjustment to theta
        self.time += 1
        for i in range(len(theta)):
            px_i = _update(theta[i], updates[i], self.eta)
            theta[i] = px_i
