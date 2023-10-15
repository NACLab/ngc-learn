"""
Mathematical/statistical functions/utilities file.
"""
import jax
import numpy as np
from jax import numpy as jnp, grad, jit, vmap, random, lax
import os, sys, pickle
from functools import partial

## general math jit-functions

def calc_acc(mu, y): ## calculates accuracy
    """
    Calculates the accuracy (ACC) given a matrix of predictions and matrix of targets.

    Args:
        mu: prediction (design) matrix

        y: target / ground-truth (design) matrix

    Returns
        scalar accuracy score
    """
    guess = jnp.argmax(mu, axis=1)
    lab = jnp.argmax(y, axis=1)
    acc = jnp.sum( jnp.equal(guess, lab) )/(y.shape[0] * 1.)
    return acc

@jit
def softmax(x, tau=0.0):
    """
    Softmax function with overflow control built in directly. Contains optional
    temperature parameter to control sharpness (tau > 1 softens probs, < 1 sharpens --> 0 yields point-mass)

    Args:
        x: a (N x D) input argument (pre-activity) to the softmax operator

        tau: probability sharpening/softening factor

    Returns:
        a (N x D) probability distribution output block
    """
    if tau > 0.0:
        x = x / tau
    max_x = jnp.max(x, axis=1, keepdims=True)
    exp_x = jnp.exp(x - max_x)
    return exp_x / jnp.sum(exp_x, axis=1, keepdims=True)

def compute_angles(W, pretty=False):
    num_vectors = W.shape[1]
    us = [W[:, i] / jnp.linalg.norm(W[:, i]) for i in range(num_vectors)]
    angles = []
    for i in range(0, num_vectors):
        for j in range(i, num_vectors):
            if i != j:
                # a = jnp.dot(us[:, i], us[:, j].T)
                a = jnp.dot(us[i], us[j].T)
                # a = a / jnp.linalg.norm(a)
                if pretty:
                    print((i, j), a, jnp.arccos(a) * 180 / math.pi)
                angles.append(a)
    return angles
