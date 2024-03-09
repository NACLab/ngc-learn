import sys, getopt, optparse
import numpy as np
from jax import jit, numpy as jnp, random, nn, lax
from functools import partial
import time

@jit
def _update(param, grad, g1, g2, lr, beta1, beta2, time, eps):
    """Parameter update using Adam.

      g1 = beta1 * g1 + (1 - beta1) * updates
      g2 = beta2 * g2 + (1 - beta2) * g2
      g1_unbiased = g1 / (1 - beta1**time)
      g2_unbiased = g2 / (1 - beta2**time)
      w = w - lr * g1_unbiased / (sqrt(g2_unbiased) + epsilon)
    """
    _g1 = beta1 * g1 + (1. - beta1) * grad
    _g2 = beta2 * g2 + (1. - beta2) * jnp.square(grad)
    g1_unb = _g1 / (1. - jnp.power(beta1, time))
    g2_unb = _g2 / (1. - jnp.power(beta2, time))
    _param = param - lr * g1_unb/(jnp.sqrt(g2_unb) + eps)
    return _param, _g1, _g2

class Adam():
    """
    Implements the adaptive moment estimation (Adam) algorithm as a decoupled
    update rule given adjustments produced by a credit assignment algorithm.

    -- Arguments --
    :param learning_rate: step size coefficient for SGD update
    :param beta1: 1st moment control factor
    :param beta2: 2nd moment control factor
    :param epsilon: numberical stability coefficient (for calculating final update)

    @author: Alexander G. Ororbia II
    """
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.eta = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = epsilon

        self.g1 = []
        self.g2 = []
        self.time = 0.

    def update(self, theta, updates):  ## apply adjustment to theta
        if self.time <= 0.: ## init statistics
            for i in range(len(theta)):
                self.g1.append(jnp.zeros(theta[i].shape))
                self.g2.append(jnp.zeros(theta[i].shape))
        self.time += 1
        for i in range(len(theta)):
            px_i, g1_i, g2_i = _update(theta[i], updates[i], self.g1[i], self.g2[i], self.eta, self.beta1, self.beta2, self.time, self.eps)
            theta[i] = px_i
            self.g1[i] = g1_i
            self.g2[i] = g2_i
