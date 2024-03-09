from ngclearn.utils.optim.opt import Opt
import numpy as np
from jax import jit, numpy as jnp, random, nn, lax
from functools import partial
import time

@jit
def step_update(param, update, g1, g2, lr, beta1, beta2, time, eps):
    """
    Runs one step of Adam over a set of parameters given updates.
    The dynamics for any set of parameters is as follows:

    | g1 = beta1 * g1 + (1 - beta1) * update
    | g2 = beta2 * g2 + (1 - beta2) * (update)^2
    | g1_unbiased = g1 / (1 - beta1**time)
    | g2_unbiased = g2 / (1 - beta2**time)
    | param = param - lr * g1_unbiased / (sqrt(g2_unbiased) + epsilon)

    Args:
        param: parameter tensor to change/adjust

        update: update tensor to be applied to parameter tensor (must be same
            shape as "param")

        g1: first moment factor/correction factor to use in parameter update
            (must be same shape as "update")

        g2: second moment factor/correction factor to use in parameter update
            (must be same shape as "update")

        lr: global step size value to be applied to updates to parameters

        beta1: 1st moment control factor

        beta2: 2nd moment control factor

        time: current time t or iteration step/call to this Adam update

        eps: numberical stability coefficient (for calculating final update)

    Returns:
        adjusted parameter tensor (same shape as "param")
    """
    _g1 = beta1 * g1 + (1. - beta1) * update
    _g2 = beta2 * g2 + (1. - beta2) * jnp.square(update)
    g1_unb = _g1 / (1. - jnp.power(beta1, time))
    g2_unb = _g2 / (1. - jnp.power(beta2, time))
    _param = param - lr * g1_unb/(jnp.sqrt(g2_unb) + eps)
    return _param, _g1, _g2

class Adam(Opt):
    """
    Implements the adaptive moment estimation (Adam) algorithm as a decoupled
    update rule given adjustments produced by a credit assignment algorithm/process.

    Args:
        learning_rate: step size coefficient for Adam update

        beta1: 1st moment control factor

        beta2: 2nd moment control factor

        epsilon: numberical stability coefficient (for calculating final update)
    """
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        super().__init__(name="adam")
        self.eta = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = epsilon

        self.g1 = []
        self.g2 = []
        #self.time = 0.

    def update(self, theta, updates):  ## apply adjustment to theta
        if self.time <= 0.: ## init statistics
            for i in range(len(theta)):
                self.g1.append(jnp.zeros(theta[i].shape))
                self.g2.append(jnp.zeros(theta[i].shape))
        self.time += 1
        for i in range(len(theta)):
            px_i, g1_i, g2_i = step_update(theta[i], updates[i], self.g1[i],
                                           self.g2[i], self.eta, self.beta1,
                                           self.beta2, self.time, self.eps)
            theta[i] = px_i
            self.g1[i] = g1_i
            self.g2[i] = g2_i
