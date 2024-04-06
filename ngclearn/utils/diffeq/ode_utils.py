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
            (supported type: rk1` or `euler`, `rk2` or `midpoint`,
            `rk2_heun` or `heun`)

    Returns:
        integator type integer code
    """
    intgFlag = 0 ## Default is Euler (RK1)
    if integrationType == "midpoint" or integrationType == "rk2": ## midpoint method
        intgFlag = 1
    elif integrationType == "rk2_heun" or integrationType == "heun": ## Heun's method
        intgFlag = 2
    # elif integrationType == "rk4": ## Runge-Kutte 4rd order code
    #     intgFlag = 3
    else:
        if integrationType != "euler" or integrationType == "rk1":
            print("ERROR: unrecognized integration method {} provided! Defaulting \
                  to RK-1/Euler routine".format(integrationType))
    return intgFlag

def step_euler(x, params, dfx, dt, dt_div=1., x_scale=1.): ## RK-1 routine
    """
    Iteratively integrates one step forward via the Euler method, i.e., a
    first-order Runge-Kutta (RK-1) step.

    Args:
        x: current variable values to advance/iteratively integrate (at time `t`)

        params: tuple containing configuration values/hyper-parameters for the
            (ordinary) differential equation an ngc-learn component will provide

        dfx: (ordinary) differential equation co-routine (as implemented in an
            ngc-learn component)

        dt: integration time step (also referred to as `h` in mathematics)

        dt_div: factor to divide `dt` by (Default: 1)

        x_scale: dampening factor to scale `x` by (Default: 1)

    Returns:
        variable values iteratively integrated/advanced to next step (`t + dt`)
    """
    dx_dt = dfx(x, params) ## assumed will be a jit-i-fied function
    return _step_forward(x, dx_dt, dt, dt_div, x_scale) ## jit-i-fied function

def step_rk2(x, params, dfx, dt): ## RK-2 routine
    """
    Iteratively integrates one step forward via the midpoint method, i.e., a
    second-order Runge-Kutta (RK-2) step. (Note: ngc-learn internally recognizes
    "rk2" or "midpoint" for this routine)

    Args:
        x: current variable values to advance/iteratively integrate (at time `t`)

        params: tuple containing configuration values/hyper-parameters for the
            (ordinary) differential equation an ngc-learn component will provide

        dfx: (ordinary) differential equation co-routine (as implemented in an
            ngc-learn component)

        dt: integration time step (also referred to as `h` in mathematics)

        dt_div: factor to divide `dt` by (Default: 1)

        x_scale: dampening factor to scale `x` by (Default: 1)

    Returns:
        variable values iteratively integrated/advanced to next step (`t + dt`)
    """
    _x1 = step_euler(x, params, dfx, dt, dt_div=2.)
    dx_dt = dfx(_x1, params) ## assumed will be a jit-i-fied function
    _x2 = _step_forward(x, dx_dt, dt) ## get 2nd order estimate
    return _x2

def step_rk2_heun(x, params, dfx, dt): ## Heun's method routine
    """
    Iteratively integrates one step forward via Heun's method, i.e., a
    second-order Runge-Kutta (RK-2) error-corrected step.
    (Note: ngc-learn internally recognizes "rk2_heun" or "heun" for this routine)

    | Reference:
    | Ascher, Uri M., and Linda R. Petzold. Computer methods for ordinary
    | differential equations and differential-algebraic equations. Society for
    | Industrial and Applied Mathematics, 1998.

    Args:
        x: current variable values to advance/iteratively integrate (at time `t`)

        params: tuple containing configuration values/hyper-parameters for the
            (ordinary) differential equation an ngc-learn component will provide

        dfx: (ordinary) differential equation co-routine (as implemented in an
            ngc-learn component)

        dt: integration time step (also referred to as `h` in mathematics)

        dt_div: factor to divide `dt` by (Default: 1)

        x_scale: dampening factor to scale `x` by (Default: 1)

    Returns:
        variable values iteratively integrated/advanced to next step (`t + dt`)
    """
    ## perform Euler estimate
    dx_dt = dfx(x, params)
    _x1 = _step_forward(x, dx_dt, dt, dt_div=1.)
    ## compute
    dx1_dt = dfx(_x1, params) ## obtain differential at intermediate step
    dx = _avg(dx_dt, dx1_dt) ## get corrected flow (sum of over/under estimates)
    _x2 = _step_forward(x, dx, dt, dt_div=1.) ## get 2nd order estimate
    return _x2

@jit
def _avg(y1, y2): ## fast co-routine for simple midpoint/average
    return (y1 + y2)/2.

@jit #@partial(jit, static_argnums=[3, 4])
def _step_forward(x, dx_dt, dt, dt_div=1., x_scale=1.): ## internal integration co-routine
    _x = x * x_scale + dx_dt * (dt/dt_div)
    return _x
