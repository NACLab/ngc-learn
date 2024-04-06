"""
Routines and co-routines for ngc-learn's differential equation integration backend.
"""
from jax import numpy as jnp, random, jit #, nn
from functools import partial
import time, sys

def get_integrator_code(integrationType):
    """
    Convenience function for mapping integrator type string to ngc-learn's
    internal integer code value.

    Args:
        integrationType: string indicating integrator type
            (`euler` or `rk1`, `rk2`)

    Returns:
        integator type integer code
    """
    intgFlag = 0 ## Default is Euler (RK1)
    if integrationType == "midpoint" or integrationType == "rk2":
        intgFlag = 1
    elif integrationType == "rk3": ## Runge-Kutte 3rd order code
        intgFlag = 2
    elif integrationType == "rk4": ## Runge-Kutte 4rd order code
        intgFlag = 3
    return intgFlag

def step_euler(x, params, dfx, dt, dt_div=1., x_scale=1.): ## RK-1 routine
    dx_dt = dfx(x, params) ## assumed will be a jit-i-fied function
    return _step_forward(x, dx_dt, dt, dt_div, x_scale) ## jit-i-fied function

def step_rk2(x, params, dfx, dt): ## RK-2 routine
    _x1 = step_euler(x, params, dfx, dt, dt_div=2.)
    dx_dt = dfx(_x1, params) ## assumed will be a jit-i-fied function
    _x2 = _step_forward(x, dx_dt, dt) ## get 2nd order estimate
    return _x2

@jit #@partial(jit, static_argnums=[3, 4])
def _step_forward(x, dx_dt, dt, dt_div=1., x_scale=1.): ## integration co-routine
    _x = x * x_scale + dx_dt * (dt/dt_div)
    return _x
