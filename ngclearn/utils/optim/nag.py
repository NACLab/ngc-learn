# %%

import numpy as np
from jax import jit, numpy as jnp, random, nn, lax
from functools import partial
import time


def step_update(param, update, phi_old, eta, mu, time_step):
    """
    Runs one step of Nesterov's accelerated gradient (NAG) over a set of parameters given updates.
    The dynamics for any set of parameters is as follows:

    | phi = param - update * lr
    | param = phi + (phi - phi_previous) * mu, where mu = 0 iff t <= 1 (first iteration)

    Args:
        param: parameter tensor to change/adjust

        update: update tensor to be applied to parameter tensor (must be same
            shape as "param")

        phi_old: previous friction/momentum parameter

        eta: global step size value to be applied to updates to parameters

        mu: friction/momentum control factor

        time_step: current time t or iteration step/call to this NAG update

    Returns:
        adjusted parameter tensor (same shape as "param"), adjusted momentum/friction variable
    """
    phi = param - update * eta ## do a phantom gradient adjustment step
    _param = phi + (phi - phi_old) * (mu * (time_step > 1.)) ## NAG-step
    _phi_old = phi
    return _param, _phi_old

@jit
def nag_step(opt_params, theta, updates, eta=0.01, mu=0.9):  ## apply adjustment to theta
    """
    Implements Nesterov's accelerated gradient (NAG) algorithm as a decoupled update rule given adjustments produced
    by a credit assignment algorithm/process.

    Args:
        opt_params: (ArrayLike) parameters of the optimization algorithm

        theta: (ArrayLike) the weights of neural network

        updates: (ArrayLike) the updates of neural network

        eta: (float, optional) step size coefficient for NAG update (Default: 0.001)

        mu: (float, optional) friction/momentum control factor. (Default: 0.9)

    Returns:
        ArrayLike: opt_params. New opt params, ArrayLike: theta. The updated weights
    """
    phi, time_step = opt_params
    time_step = time_step + 1
    new_theta = []
    new_phi = []
    for i in range(len(theta)):
        px_i, phi_i = step_update(theta[i], updates[i], phi[i], eta, mu, time_step)
        new_theta.append(px_i)
        new_phi.append(phi_i)
    return (new_phi, time_step), new_theta

@jit
def nag_init(theta):
    time_step = jnp.asarray(0.0)
    phi = [jnp.zeros(theta[i].shape) for i in range(len(theta))]
    return phi, time_step

if __name__ == '__main__':
    weights = [jnp.asarray([3.0, 3.0]), jnp.asarray([3.0, 3.0])]
    updates = [jnp.asarray([3.0, 3.0]), jnp.asarray([3.0, 3.0])]
    opt_params = nag_init(weights)
    opt_params, theta = nag_step(opt_params, weights, updates)
    print(f"opt_params: {opt_params}, theta: {theta}")
    weights = theta
    print("##################")
    opt_params, theta = nag_step(opt_params, weights, updates)
    print(f"opt_params: {opt_params}, theta: {theta}")
