from ngcsimlib.component import Component
from ngcsimlib.compartment import Compartment
from ngcsimlib.resolver import resolver
from jax import numpy as jnp, random, jit
from functools import partial
import time
from ngclearn.utils.diffeq.ode_utils import get_integrator_code, \
                                            step_euler, step_rk2

import sys

@jit
def update_times(t, s, tols):
    """
    Updates time-of-last-spike (tols) variable.

    Args:
        t: current time (a scalar/int value)

        s: binary spike vector

        tols: current time-of-last-spike variable

    Returns:
        updated tols variable
    """
    _tols = (1. - s) * tols + (s * t)
    return _tols

@jit
def _dfv_internal(j, v, w, tau_m, v_rest, sharpV, vT, R_m): ## raw voltage dynamics
    dv_dt = -(v - v_rest) + sharpV * jnp.exp((v - vT)/sharpV) - R_m * w + R_m * j ## dv/dt
    dv_dt = dv_dt * (1./tau_m)
    return dv_dt

def _dfv(t, v, params): ## voltage dynamics wrapper
    j, w, tau_m, v_rest, sharpV, vT, R_m = params
    dv_dt = _dfv_internal(j, v, w, tau_m, v_rest, sharpV, vT, R_m)
    return dv_dt

@jit
def _dfw_internal(j, v, w, a, tau_w, v_rest): ## raw recovery dynamics
    dw_dt = -w + (v - v_rest) * a #+ b * s * tau_w
    dw_dt = dw_dt * (1./tau_w)
    return dw_dt

def _dfw(t, w, params): ## recovery dynamics wrapper
    j, v, a, tau_m, v_rest = params
    dv_dt = _dfw_internal(j, v, w, a, tau_m, v_rest)
    return dv_dt

@jit
def _emit_spike(v, v_thr):
    s = (v > v_thr).astype(jnp.float32)
    return s

#@partial(jit, static_argnums=[10])
def run_cell(dt, j, v, w, v_thr, tau_m, tau_w, a, b, sharpV, vT,
             v_rest, v_reset, R_m, integType=0):
    if integType == 1: ## RK-2/midpoint
        v_params = (j, w, tau_m, v_rest, sharpV, vT, R_m)
        _, _v = step_rk2(0., v, _dfv, dt, v_params)
        w_params = (j, v, a, tau_w, v_rest)
        _, _w = step_rk2(0., w, _dfw, dt, w_params)
    else: # integType == 0 (default -- Euler)
        v_params = (j, w, tau_m, v_rest, sharpV, vT, R_m)
        _, _v = step_euler(0., v, _dfv, dt, v_params)
        w_params = (j, v, a, tau_w, v_rest)
        _, _w = step_euler(0., w, _dfw, dt, w_params)
    #s = (_v > v_thr).astype(jnp.float32)
    s = _emit_spike(_v, v_thr)
    ## hyperpolarize/reset/snap variables
    _v = _v * (1. - s) + s * v_reset
    _w = _w * (1. - s) + s * (_w + b)
    return _v, _w, s

class AdExCell(Component):
    """
    UNTESTED

    The AdEx (adaptive exponential leaky integrate-and-fire) neuronal cell
    model; a two-variable model. This cell model iteratively evolves
    voltage "v" and recovery "w".

    The specific pair of differential equations that characterize this cell
    are (for adjusting v and w, given current j, over time):

    | XXX
    | YYY

    | References:
    | XXXX

    Args:
        name: the string name of this cell

        n_units: number of cellular entities (neural population size)

        tau_m: membrane time constant

        tau_w: recover variable time constant (Default: 12.5 ms)

        alpha: dimensionless recovery variable shift factor "a" (Default: 0.7)

        beta: dimensionless recovery variable scale factor "b" (Default: 0.8)

        gamma: power-term divisor (Default: 3.)

        v_thr: voltage/membrane threshold (to obtain action potentials in terms
            of binary spikes)

        v0: initial condition / reset for voltage

        w0: initial condition / reset for recovery

        integration_type: type of integration to use for this cell's dynamics;
            current supported forms include "euler" (Euler/RK-1 integration)
            and "midpoint" or "rk2" (midpoint method/RK-2 integration) (Default: "euler")

            :Note: setting the integration type to the midpoint method will
                increase the accuray of the estimate of the cell's evolution
                at an increase in computational cost (and simulation time)

        key: PRNG key to control determinism of any underlying synapses
            associated with this cell

        useVerboseDict: triggers slower, verbose dictionary mode (Default: False)
    """

    # Define Functions
    def __init__(self, name, n_units, tau_m=15., R_m=1., tau_w=400.,
                 sharpV=2., vT=-55., v_thr=5., v_rest=-72., v_reset=-75.,
                 a=0.1, b=0.75, v0=-70., w0=0.,
                 integration_type="euler", key=None, useVerboseDict=False,
                 **kwargs):
        super().__init__(name, useVerboseDict, **kwargs)

        ## Random Number Set up
        self.key = key
        if self.key is None:
            self.key = random.PRNGKey(time.time_ns())

        ## Integration properties
        self.integrationType = integration_type
        self.intgFlag = get_integrator_code(self.integrationType)

        ## Cell properties
        self.tau_m = tau_m
        self.R_m = R_m
        self.tau_w = tau_w
        self.sharpV = sharpV ## sharpness of action potential
        self.vT = vT ## intrinsic membrane threshold
        self.a = a
        self.b = b
        self.v_rest = v_rest
        self.v_reset = v_reset

        self.v0 = v0 ## initial membrane potential/voltage condition
        self.w0 = w0 ## initial w-parameter condition
        self.v_thr = v_thr

        ## Layer Size Setup
        self.batch_size = 1
        self.n_units = n_units

        ## Compartment setup
        restVals = jnp.zeros((self.batch_size, self.n_units))
        self.j = Compartment(restVals)
        self.v = Compartment(restVals + self.v0)
        self.w = Compartment(restVals + self.w0)
        self.s = Compartment(restVals)
        self.tols = Compartment(restVals) ## time-of-last-spike
        self.key = Compartment(random.PRNGKey(time.time_ns()) if key is None else key)

        #self.reset()

    @staticmethod
    def pure_advance(t, dt, tau_m, R_m, tau_w, v_thr, a, b, sharpV, vT,
                     v_rest, v_reset, intgFlag, key, j, v, w, s, tols):
        key, *subkeys = random.split(key, 2)
        v, w, s = run_cell(dt, j, v, w, v_thr, tau_m, tau_w, a, b, sharpV, vT,
                           v_rest, v_reset, R_m, intgFlag)
        tols = update_times(t, s, tols)
        return j, v, w, s, tols, key

    @resolver(pure_advance, output_compartments=['j', 'v', 'w', 's', 'tols', 'key'])
    def advance(self, vals):
        j, v, w, s, tols, key = vals
        self.j.set(j)
        self.w.set(w)
        self.v.set(v)
        self.s.set(s)
        self.tols.set(tols)
        self.key.set(key)

    @staticmethod
    def pure_reset(batch_size, n_units, v0, w0):
        restVals = jnp.zeros((batch_size, n_units))
        j = restVals # None
        v = restVals + v0
        w = restVals + w0
        s = restVals #+ 0
        tols = restVals #+ 0
        return j, v, w, s, tols

    @resolver(pure_reset, output_compartments=['j', 'v', 'w', 's', 'tols'])
    def reset(self, vals):
        j, v, w, s, tols = vals
        self.j.set(j)
        self.v.set(v)
        self.w.set(w)
        self.s.set(s)
        self.tols.set(tols)

    def save(self, **kwargs):
        pass

    # def verify_connections(self):
    #     pass
