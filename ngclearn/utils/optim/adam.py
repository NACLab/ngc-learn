# %%

import numpy as np
from jax import jit, numpy as jnp, random, nn, lax
from functools import partial


def step_update(param, update, g1, g2, eta, beta1, beta2, time_step, eps):
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

        eta: global step size value to be applied to updates to parameters

        beta1: 1st moment control factor

        beta2: 2nd moment control factor

        time_step: current time t or iteration step/call to this Adam update

        eps: numberical stability coefficient (for calculating final update)

    Returns:
        adjusted parameter tensor (same shape as "param"), adjusted g1, adjusted g2
    """
    _g1 = beta1 * g1 + (1. - beta1) * update
    _g2 = beta2 * g2 + (1. - beta2) * jnp.square(update)
    g1_unb = _g1 / (1. - jnp.power(beta1, time_step))
    g2_unb = _g2 / (1. - jnp.power(beta2, time_step))
    _param = param - eta * g1_unb/(jnp.sqrt(g2_unb) + eps)
    return _param, _g1, _g2

@jit
def adam_step(opt_params, theta, updates, eta=0.001, beta1=0.9, beta2=0.999, eps=1e-8):  ## apply adjustment to theta
    """Implements the adaptive moment estimation (Adam) algorithm as a decoupled
        update rule given adjustments produced by a credit assignment algorithm/process.

    Args:
        opt_params: (ArrayLike) parameters of the optimization algorithm

        theta: (ArrayLike) the weights of neural network

        updates: (ArrayLike) the updates of neural network

        eta: (float, optional) step size coefficient for Adam update (Default: 0.001)

        beta1: (float, optional) 1st moment control factor. (Default: 0.9)

        beta2: (float, optional) 2nd moment control factor. (Default: 0.999)

        eps: (float, optional) numberical stability coefficient (for calculating
            final update). (Default: 1e-8)

    Returns:
        ArrayLike: opt_params. New opt params, ArrayLike: theta. The updated weights
    """
    g1, g2, time_step = opt_params
    time_step = time_step + 1
    new_theta = []
    new_g1 = []
    new_g2 = []
    for i in range(len(theta)):
        px_i, g1_i, g2_i = step_update(theta[i], updates[i], g1[i], g2[i], eta, beta1, beta2, time_step, eps)
        new_theta.append(px_i)
        new_g1.append(g1_i)
        new_g2.append(g2_i)
    return (new_g1, new_g2, time_step), new_theta

@jit
def adam_init(theta):
    time_step = jnp.asarray(0.0)
    g1 = [jnp.zeros(theta[i].shape) for i in range(len(theta))]
    g2 = [jnp.zeros(theta[i].shape) for i in range(len(theta))]
    return g1, g2, time_step

if __name__ == '__main__':
    weights = [jnp.asarray([3.0, 3.0]), jnp.asarray([3.0, 3.0])]
    updates = [jnp.asarray([3.0, 3.0]), jnp.asarray([3.0, 3.0])]
    opt_params = adam_init(weights)
    opt_params, theta = adam_step(opt_params, weights, updates)
    print(f"opt_params: {opt_params}, theta: {theta}")
