"""
A builder sub-package for spike emission functions that are mapped to
surrogate (derivative) functions; these function builders are useful if
differentiation through the discrete spike emission steps in spiking neuronal
cells is required (e.g., cases of surrogate backprop,
broadcast feedback alignment schemes, etc.). Calling the builder estimator
functions below returns two routines:
1) a spike emission routine `spike(v, v_thr)`, 2) its corresponding surrogate
derivative routine `spike(j, v, v_thr, params)`.
"""
from jax import numpy as jnp, random, jit
from functools import partial
import time, sys

def straight_through_estimator():
    """
    The straight-through estimator (STE) applied to binary spike emission
    (the Heaviside function).

    | Bengio, Yoshua, Nicholas Léonard, and Aaron Courville. "Estimating or
    | propagating gradients through stochastic neurons for conditional
    | computation." arXiv preprint arXiv:1308.3432 (2013).

    Returns:
        spike_fx(x), d_spike_fx(x)
    """
    @jit
    def spike_fx(v, thr):
        return (v > thr).astype(jnp.float32)
    @jit
    def d_spike_fx(j, v, thr):
        return v * 0 + 1.
    return spike_fx, d_spike_fx

def triangular_estimator():
    """
    The triangular surrogate gradient estimator for binary spike emission.

    Returns:
        spike_fx(x), d_spike_fx(x)
    """
    @jit
    def spike_fx(v, thr):
        return (v > thr).astype(jnp.float32)
    @jit
    def d_spike_fx(j, v, thr):
        mask = (v < v_thr).astype(jnp.float32)
        dfx = mask * v_thr - (1. - mask) * v_thr
        return dfx
    return spike_fx, d_spike_fx

def secant_lif_estimator():
    """
    Surrogate function for computing derivative of (binary) spike function
    with respect to the input electrical current/drive to a leaky
    integrate-and-fire (LIF) neuron. (Note this is only useful for
    LIF neuronal dynamics.)

    | spike_fx(x) ~ E(x) = sech(x) = 1/cosh(x), cosh(x) = (e^x + e^(-x))/2
    | dE(x)/dj = (c1 c2) * sech^2(c2 * j) for j > 0 and 0 for j <= 0

    | Reference:
    | Samadi, Arash, Timothy P. Lillicrap, and Douglas B. Tweed. "Deep learning with
    | dynamic spiking neurons and fixed feedback weights." Neural computation 29.3
    | (2017): 578-602.

    Returns:
        spike_fx(x), d_spike_fx(x)
    """
    @jit
    def spike_fx(v, thr):
        #return jnp.where(new_voltage > v_thr, 1, 0)
        return (v > thr).astype(jnp.float32)
    @partial(jit, static_argnums=[5])
    def d_spike_fx(j, v, thr, c1=0.82, c2=0.08, omit_scale=True): #c1=0.82, c2=0.08):
        """
        | dE(x)/dj = scale * sech^2(c2 * j) for j > 0 and 0 for j <= 0;
        | where scale = (c1 * c2) if `omit_scale = False`, otherwise, scale = 1.

        Args:
            j: electrical current value

            v: voltage (unused)

            thr: voltage threshold (unused)

            c1: control coefficient 1 (unnamed factor from paper - scales current
                input; Default: 0.82 as in source paper)

            c2: control coefficient 2 (unnamed factor from paper - scales, multiplicatively
                with c1, the output the derivative surrogate; Default: 0.08 as in
                source paper)

            omit_scale: preserves final scaling of dv_dj by (c1 * c2) if False and
                (Default: True)

        Returns:
            surrogate output values (same shape as j)
        """
        mask = (j > 0.).astype(jnp.float32)
        dj = j * c2
        cosh_j = (jnp.exp(dj) + jnp.exp(-dj))/2.
        sech_j = 1./cosh_j #(cosh_x + 1e-6)
        dv_dj = sech_j #* (c1 * c2) # ~deriv w.r.t. j
        if omit_scale == False:
            dv_dj = dv_dj * (c1 * c2)
        return dv_dj * mask ## 0 if j < 0, otherwise, use dv/dj for j >= 0
    return spike_fx, d_spike_fx
