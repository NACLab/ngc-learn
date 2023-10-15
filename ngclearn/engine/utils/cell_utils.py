"""
Cell utility function file - includes spike neuron models, threshold functions, etc.
"""

import jax
import numpy as np
from jax import numpy as jnp, grad, jit, vmap, random, lax
import os, sys, pickle
from functools import partial

@jit
def calc_snm_bool(v, v_thr):
    ## boolean-comparison based spike model
    s = (v > v_thr).astype(jnp.float32)
    return s

@jit
def calc_snm_wta(v, v_thr):
    ## winner-take-all / argmax based spike model
    s = nn.one_hot(jnp.argmax(v, axis=1), j.shape[1])
    return s

@jit
def calc_global_thr(s, v_thr, thr_gain, thr_decay=0., q=1., vthr_min=0.001): # vthr_min=0.05
    ## Global dynamic threshold update function
    ## Note: global thresholds must be non-persistent
    d_v_thr = (jnp.sum(s, axis=1, keepdims=True) - q) * thr_gain
    _v_thr = v_thr + d_v_thr
    _v_thr = jnp.fmax(_v_thr, vthr_min)
    return _v_thr

@jit
def calc_percell_thr(s, v_thr, thr_gain, thr_decay, vthr_min=0.05):
    ## Persistent threshold update function
    d_v_thr = (s * thr_gain) - (v_thr * thr_decay)
    _v_thr = v_thr + d_v_thr
    return _v_thr

@jit
def calc_persist_percell_thr(s, v_thr, thr_gain, thr_decay, vthr_min=0.05):
    ## Dynamic threshold update function
    d_v_thr = (s * thr_gain) - (v_thr * thr_decay)
    d_v_thr_mu = jnp.mean(d_v_thr, axis=0, keepdims=True)
    _v_thr = v_thr  + d_v_thr_mu
    return _v_thr

@jit
def apply_lat_pressure(j, s_tm1, Vmat, Rinh):
    ## A lateral inhibitory pressure convenience function (for electrical currents)
    j_inh = jnp.matmul(s_tm1, Vmat) * Rinh
    return j - j_inh, j_inh # inhibitory current is subtracted from j
