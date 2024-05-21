# %%

from ngcsimlib.component import Component
from ngcsimlib.compartment import Compartment
from ngcsimlib.resolver import resolver

import numpy as np
from jax import jit, numpy as jnp, random, nn, lax
from functools import partial
import time

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

@jit
def sgd_step(opt_params, theta, updates, eta=0.001): ## apply adjustment to theta
    """Return a params update

    Args:
        opt_params: (ArrayLike) parameters of the optimization algorithm

        theta: (ArrayLike) the weights of neural networks

        updates: (ArrayLike) the updates of neural networks

        eta: (float, optional) hyperparams. Defaults to 0.001.

    Returns:
        ArrayLike: opt_params. New opt params, ArrayLike: theta. The updated weights
    """
    time_step = opt_params
    time_step = time_step + 1
    new_theta = []
    for i in range(len(theta)):
        px_i = step_update(theta[i], updates[i], eta)
        new_theta.append(px_i)
    new_opt_params = time_step
    return new_opt_params, new_theta

@jit
def sgd_init(theta):
    return jnp.asarray(0.0)


if __name__ == '__main__':
    opt_params, theta = sgd_step((2.0), [1.0, 1.0], [3.0, 4.0], 3e-2)
    print(f"opt_params: {opt_params}, theta: {theta}")
