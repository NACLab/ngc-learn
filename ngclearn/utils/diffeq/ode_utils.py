"""
Routines and co-routines for ngc-learn's differential equation integration backend.
"""
from jax import numpy as jnp, random, jit #, nn
from functools import partial
import time, sys

def step_euler(x, params, dfx, dt): ## RK-1 routine
    dx_dt = dfx(x, params) ## assumed will be a jit-i-fied function
    return _step_forward(x, dx_dt, dt) ## jit-i-fied function

def step_rk2(x, params, dfx, dt): ## RK-2 routine
    _x1 = step_euler(x, params, dfx, dt, dt_div=2.)
    dx_dt = dfx(_x1, params) ## assumed will be a jit-i-fied function
    return _step_forward(x, dx_dt, dt) ## get 2nd order estimate

@partial(jit, static_argnums=[3, 4])
def _step_forward(x, dx_dt, dt, dt_div=1., x_scale=1.): ## integration co-routine
    _x = x * x_scale + dx_dt * (dt/dt_div)
    return _x
