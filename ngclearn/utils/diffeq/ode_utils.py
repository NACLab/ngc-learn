"""
Routines and co-routines for ngc-learn's differential equation integration backend.

Currently supported back-end forms of integration in ngc-learn include:
0) Euler integration (RK-1);
1) Midpoint method (RK-2);
2) Heun's method (error-corrector RK-2);
3) Ralston's method (error-corrector RK-2);
4) 4th-order Runge-Kutta method (RK-4);
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
            `rk2_heun` or `heun`, `rk2_ralston` or `ralston`, `rk4`)

    Returns:
        integator type integer code
    """
    intgFlag = 0 ## Default is Euler (RK1)
    if integrationType == "midpoint" or integrationType == "rk2": ## midpoint method
        intgFlag = 1
    elif integrationType == "rk2_heun" or integrationType == "heun": ## Heun's method
        intgFlag = 2
    elif integrationType == "rk2_ralston" or integrationType == "ralston": ## Ralston's method
        intgFlag = 3
    elif integrationType == "rk4": ## Runge-Kutte 4rd order code
        intgFlag = 4
    else:
        if integrationType != "euler" or integrationType == "rk1":
            print("ERROR: unrecognized integration method {} provided! Defaulting \
                  to RK-1/Euler routine".format(integrationType))
    return intgFlag


@jit
def _sum_combine(*args, **kwargs): ## fast co-routine for simple addition
    sum = 0
    for arg, val in zip(args, kwargs.values()):
        sum += arg * val
    return sum

@jit
def _step_forward(t, x, dx_dt, dt, x_scale): ## internal step co-routine
    _t = t + dt
    _x = x * x_scale + dx_dt * dt
    return _t, _x

def step_euler(t, x, dfx, dt, params, x_scale=1.):
    """
    Iteratively integrates one step forward via the Euler method, i.e., a
    first-order Runge-Kutta (RK-1) step.

    Args:
        t: current time variable to advance by dt

        x: current variable values to advance/iteratively integrate (at time `t`)

        dfx: (ordinary) differential equation co-routine (as implemented in an
            ngc-learn component)

        dt: integration time step (also referred to as `h` in mathematics)

        params: tuple containing configuration values/hyper-parameters for the
            (ordinary) differential equation an ngc-learn component will provide

        x_scale: dampening factor to scale `x` by (Default: 1)

    Returns:
        variable values iteratively integrated/advanced to next step (`t + dt`)
    """
    dx_dt = dfx(t, x, params)
    _t, _x = _step_forward(t, x, dx_dt, dt, x_scale)
    return _t, _x

def step_heun(t, x, dfx, dt, params, x_scale=1.):
    """
    Iteratively integrates one step forward via Heun's method, i.e., a
    second-order Runge-Kutta (RK-2) error-corrected step. This method utilizes
    two (differential) function evaluations to estimate the solution at a given
    point in time.
    (Note: ngc-learn internally recognizes "rk2_heun" or "heun" for this routine)

    | Reference:
    | Ascher, Uri M., and Linda R. Petzold. Computer methods for ordinary
    | differential equations and differential-algebraic equations. Society for
    | Industrial and Applied Mathematics, 1998.

    Args:
        t: current time variable to advance by dt

        x: current variable values to advance/iteratively integrate (at time `t`)

        dfx: (ordinary) differential equation co-routine (as implemented in an
            ngc-learn component)

        dt: integration time step (also referred to as `h` in mathematics)

        params: tuple containing configuration values/hyper-parameters for the
            (ordinary) differential equation an ngc-learn component will provide

        x_scale: dampening factor to scale `x` by (Default: 1)

    Returns:
        variable values iteratively integrated/advanced to next step (`t + dt`)
    """
    dx_dt = dfx(t, x, params)
    _t, _x = _step_forward(t, x, dx_dt, dt, x_scale)
    _dx_dt = dfx(_t, _x, params)
    summed_dx_dt = _sum_combine(dx_dt, _dx_dt, weight1=1, weight2=1)
    _, _x = _step_forward(t, x, summed_dx_dt, dt * 0.5, x_scale)
    return _t, _x


def step_rk2(t, x, dfx, dt, params, x_scale=1.):
    """
    Iteratively integrates one step forward via the midpoint method, i.e., a
    second-order Runge-Kutta (RK-2) step.
    (Note: ngc-learn internally recognizes "rk2" or "midpoint" for this routine)

    | Reference:
    | Ascher, Uri M., and Linda R. Petzold. Computer methods for ordinary
    | differential equations and differential-algebraic equations. Society for
    | Industrial and Applied Mathematics, 1998.

    Args:
        t: current time variable to advance by dt

        x: current variable values to advance/iteratively integrate (at time `t`)

        dfx: (ordinary) differential equation co-routine (as implemented in an
            ngc-learn component)

        dt: integration time step (also referred to as `h` in mathematics)

        params: tuple containing configuration values/hyper-parameters for the
            (ordinary) differential equation an ngc-learn component will provide

        x_scale: dampening factor to scale `x` by (Default: 1)

    Returns:
        variable values iteratively integrated/advanced to next step (`t + dt`)
    """
    f_1 = dfx(t, x, params)
    t1, x1 = _step_forward(t, x, f_1, dt * 0.5, x_scale)

    f_2 = dfx(t1, x1, params)
    _t, _x = _step_forward(t, x, f_2, dt, x_scale)
    return _t, _x


def step_rk4(t, x, dfx, dt, params, x_scale=1.):
    """
    Iteratively integrates one step forward via the midpoint method, i.e., a
    fourth-order Runge-Kutta (RK-4) step.
    (Note: ngc-learn internally recognizes "rk4" or this routine)

    | Reference:
    | Ascher, Uri M., and Linda R. Petzold. Computer methods for ordinary
    | differential equations and differential-algebraic equations. Society for
    | Industrial and Applied Mathematics, 1998.

    Args:
        t: current time variable to advance by dt

        x: current variable values to advance/iteratively integrate (at time `t`)

        dfx: (ordinary) differential equation co-routine (as implemented in an
            ngc-learn component)

        dt: integration time step (also referred to as `h` in mathematics)

        params: tuple containing configuration values/hyper-parameters for the
            (ordinary) differential equation an ngc-learn component will provide

        x_scale: dampening factor to scale `x` by (Default: 1)

    Returns:
        variable values iteratively integrated/advanced to next step (`t + dt`)
    """
    f_1 = dfx(t, x, params)
    t1, x1 = _step_forward(t, x, f_1, dt * 0.5, x_scale)

    f_2 = dfx(t1, x1, params)
    t2, x2 = _step_forward(t, x, f_2, dt * 0.5, x_scale)

    f_3 = dfx(t2, x2, params)
    t3, x3 = _step_forward(t, x, f_3, dt, x_scale)

    f_4 = dfx(t3, x3, params)

    _dx_dt = _sum_combine(f_1, f_2, f_3, f_4, w_f1=1, w_f2=2, w_f3=2, w_f4=1)
    _t, _x = _step_forward(t, x, _dx_dt, dt / 6, x_scale)
    return _t, _x



def step_ralston(t, x, dfx, dt, params, x_scale=1.):
    """
    Iteratively integrates one step forward via Ralston's method, i.e., a
    second-order Runge-Kutta (RK-2) error-corrected step. This method utilizes
    two (differential) function evaluations to estimate the solution at a given
    point in time.
    (Note: ngc-learn internally recognizes "rk2_ralston" or "ralston" for this
    routine)

    | Reference:
    | Ralston, Anthony. "Runge-Kutta methods with minimum error bounds."
    | Mathematics of computation 16.80 (1962): 431-437.

    Args:
        t: current time variable to advance by dt

        x: current variable values to advance/iteratively integrate (at time `t`)

        dfx: (ordinary) differential equation co-routine (as implemented in an
            ngc-learn component)

        dt: integration time step (also referred to as `h` in mathematics)

        params: tuple containing configuration values/hyper-parameters for the
            (ordinary) differential equation an ngc-learn component will provide

        x_scale: dampening factor to scale `x` by (Default: 1)

    Returns:
        variable values iteratively integrated/advanced to next step (`t + dt`)
    """
    dx_dt = dfx(t, x, params) ## k1
    tm, xm = _step_forward(t, x, dx_dt, dt * 0.75, x_scale)
    _dx_dt = dfx(tm, xm, params)  ## k2
    ## Note: new step is a weighted combination of k1 and k2
    summed_dx_dt = _sum_combine(dx_dt, _dx_dt, weight1=(1./3.), weight2=(2./3.))
    _t, _x = _step_forward(t, x, summed_dx_dt, dt, x_scale)
    return _t, _x
