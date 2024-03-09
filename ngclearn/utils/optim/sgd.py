from ngclearn.utils.optim.opt import Opt
import numpy as np
from jax import jit, numpy as jnp, random, nn, lax
from functools import partial
import time

@jit
def step_update(param, update, lr):
    """
    Runs one step of SGD over a set of parameters given updates.

    Args:
        lr: global step size to apply when adjusting parameters

    Returns:
        adjusted parameter tensor (same shape as "param")
    """
    _param = param - lr * update
    return _param

class SGD(Opt):
    """
    Implements stochastic gradient descent (SGD) as a decoupled update rule
    given adjustments produced by a credit assignment algorithm/process.

    Args:
        learning_rate: step size coefficient for SGD update
    """
    def __init__(self, learning_rate=0.001):
        super().__init__(name="sgd")
        self.eta = learning_rate
        #self.time = 0.

    def update(self, theta, updates): ## apply adjustment to theta
        self.time += 1
        for i in range(len(theta)):
            px_i = step_update(theta[i], updates[i], self.eta)
            theta[i] = px_i
