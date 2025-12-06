"""
Routines and co-routines for ngc-learn's differential equation integration backend.

| Currently supported back-end forms of integration in ngc-learn include:
| 0) Euler integration (RK-1);
| 1) Midpoint method (RK-2);
| 2) Heun's method (error-corrector RK-2);
| 3) Ralston's method (error-corrector RK-2);
| 4) 4th-order Runge-Kutta method (RK-4);

"""

from jax import numpy as jnp, random, jit #, nn
from functools import partial
from jax.lax import scan as _scan
import time, sys

def get_integrator_code(integrationType): ## integrator type decoding routine
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
def _sum_combine(*args, **kwargs): ## fast co-routine for simple addition/summation
    _sum = 0
    for arg, val in zip(args, kwargs.values()): ## Sigma^I_{i=1} a_i
        _sum = _sum + val * arg
    return _sum

@jit
def _step_forward(t, x, dx_dt, dt, x_scale): ## internal step co-routine
    _t = t + dt ## advance time forward by dt (denominator)
    _x = x * x_scale + dx_dt * dt ## advance variable(s) forward by dt (numerator)
    return _t, _x

@partial(jit, static_argnums=(2))
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
    carry = (t, x)
    next_state, *_ = _euler(carry, dfx, dt, params, x_scale=x_scale)
    _t, _x = next_state
    return _t, _x

@partial(jit, static_argnums=(1))
def _euler(carry, dfx, dt, params, x_scale=1.):
    """
    Iteratively integrates one step forward via the Euler method, i.e., a
    first-order Runge-Kutta (RK-1) step.

    Args:
        carry: a tuple containing current time and data, i.e., (t, x)

        dfx: (ordinary) differential equation co-routine (as implemented in an
            ngc-learn component)

        dt: integration time step (also referred to as `h` in mathematics)

        params: tuple containing configuration values/hyper-parameters for the
            (ordinary) differential equation an ngc-learn component will provide

        x_scale: dampening factor to scale `x` by (Default: 1)

    Returns:
        variable values iteratively integrated/advanced to next step (`t + dt`)
    """
    t, x = carry
    dx_dt = dfx(t, x, params)
    _t, _x = _step_forward(t, x, dx_dt, dt, x_scale)
    new_carry = (_t, _x)
    return new_carry, (new_carry, carry)

@partial(jit, static_argnums=(1))
def _leapfrog(carry, dfq, dt, params):
    t, q, p = carry
    dq_dt = dfq(t, q, params)

    _p = p + dq_dt * (dt/2.)
    _q = q + p * dt
    dq_dtpdt = dfq(t+dt, _q, params)
    _p = _p + dq_dtpdt * (dt/2.)
    _t = t + dt
    new_carry = (_t, _q, _p)
    return new_carry, (new_carry, carry)

@partial(jit, static_argnums=(3, 4))
def leapfrog(t_curr, q_curr, p_curr, dfq, L, step_size, params):
    t = t_curr + 0.
    q = q_curr + 0.
    p = p_curr + 0.
    def scanner(carry, _):
        return _leapfrog(carry, dfq, step_size, params)
    new_values, (xs_next, xs_carry) = _scan(scanner, init=(t, q, p), xs=jnp.arange(L))
    t, q, p = new_values
    return t, q, p

@partial(jit, static_argnums=(2))
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

    carry = (t, x)
    next_state, *_ = _heun(carry, dfx, dt, params, x_scale=x_scale)
    _t, _x = next_state
    return _t, _x

@partial(jit, static_argnums=(1))
def _heun(carry, dfx, dt, params, x_scale=1.):
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
        carry: a tuple containing current time and data, i.e., (t, x)

        dfx: (ordinary) differential equation co-routine (as implemented in an
            ngc-learn component)

        dt: integration time step (also referred to as `h` in mathematics)

        params: tuple containing configuration values/hyper-parameters for the
            (ordinary) differential equation an ngc-learn component will provide

        x_scale: dampening factor to scale `x` by (Default: 1)

    Returns:
        variable values iteratively integrated/advanced to next step (`t + dt`)
    """
    t, x = carry
    dx_dt = dfx(t, x, params)
    _t, _x = _step_forward(t, x, dx_dt, dt, x_scale)
    _dx_dt = dfx(_t, _x, params)
    summed_dx_dt = _sum_combine(dx_dt, _dx_dt, weight1=1, weight2=1)
    _, _x = _step_forward(t, x, summed_dx_dt, dt * 0.5, x_scale)
    new_carry = (_t, _x)
    return new_carry, (new_carry, carry)

@partial(jit, static_argnums=(2))
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
    carry = (t, x)
    next_state, *_ = _rk2(carry, dfx, dt, params, x_scale=x_scale)
    _t, _x = next_state
    return _t, _x

@partial(jit, static_argnums=(1))
def _rk2(carry, dfx, dt, params, x_scale=1.):
    """
    Iteratively integrates one step forward via the midpoint method, i.e., a
    second-order Runge-Kutta (RK-2) step.
    (Note: ngc-learn internally recognizes "rk2" or "midpoint" for this routine)

    | Reference:
    | Ascher, Uri M., and Linda R. Petzold. Computer methods for ordinary
    | differential equations and differential-algebraic equations. Society for
    | Industrial and Applied Mathematics, 1998.

    Args:
        carry: a tuple containing current time and data, i.e., (t, x)

        dfx: (ordinary) differential equation co-routine (as implemented in an
            ngc-learn component)

        dt: integration time step (also referred to as `h` in mathematics)

        params: tuple containing configuration values/hyper-parameters for the
            (ordinary) differential equation an ngc-learn component will provide

        x_scale: dampening factor to scale `x` by (Default: 1)

    Returns:
        variable values iteratively integrated/advanced to next step (`t + dt`)
    """
    t, x = carry
    f_1 = dfx(t, x, params)
    t1, x1 = _step_forward(t, x, f_1, dt * 0.5, x_scale)
    f_2 = dfx(t1, x1, params)
    _t, _x = _step_forward(t, x, f_2, dt, x_scale)
    new_carry = (_t, _x)
    return new_carry, (new_carry, carry)

@partial(jit, static_argnums=(2))
def step_rk4(t, x, dfx, dt, params, x_scale=1.):
    """
    Iteratively integrates one step forward via the midpoint method, i.e., a
    fourth-order Runge-Kutta (RK-4) step.
    (Note: ngc-learn internally recognizes "rk4" for this routine)

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
    carry = (t, x)
    next_state, *_ = _rk4(carry, dfx, dt, params, x_scale=x_scale)
    _t, _x = next_state
    return _t, _x

@partial(jit, static_argnums=(1))
def _rk4(carry, dfx, dt, params, x_scale=1.):
    """
    Iteratively integrates one step forward via the midpoint method, i.e., a
    fourth-order Runge-Kutta (RK-4) step.
    (Note: ngc-learn internally recognizes "rk4" or this routine)

    | Reference:
    | Ascher, Uri M., and Linda R. Petzold. Computer methods for ordinary
    | differential equations and differential-algebraic equations. Society for
    | Industrial and Applied Mathematics, 1998.

    Args:
        carry: a tuple containing current time and data, i.e., (t, x)

        dfx: (ordinary) differential equation co-routine (as implemented in an
            ngc-learn component)

        dt: integration time step (also referred to as `h` in mathematics)

        params: tuple containing configuration values/hyper-parameters for the
            (ordinary) differential equation an ngc-learn component will provide

        x_scale: dampening factor to scale `x` by (Default: 1)

    Returns:
        variable values iteratively integrated/advanced to next step (`t + dt`)
    """
    t, x = carry
    ## carry out 4 steps of RK-4
    dfx_1 = dfx(t, x, params) ## k1
    t2, x2 = _step_forward(t, x, dfx_1, dt * 0.5, x_scale)
    dfx_2 = dfx(t2, x2, params) ## k2
    t3, x3 = _step_forward(t, x, dfx_2, dt * 0.5, x_scale)
    dfx_3 = dfx(t3, x3, params) ## k3
    t4, x4 = _step_forward(t, x, dfx_3, dt, x_scale)
    dfx_4 = dfx(t4, x4, params) ## k4
    ## produce final estimate and move forward
    _dx_dt = _sum_combine(dfx_1, dfx_2, dfx_3, dfx_4, w_f1=1, w_f2=2, w_f3=2, w_f4=1)
    _t, _x = _step_forward(t, x, _dx_dt / 6, dt, x_scale)
    new_carry = (_t, _x)
    return new_carry, (new_carry, carry)

@partial(jit, static_argnums=(2))
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
    carry = (t, x)
    next_state, *_ = _ralston(carry, dfx, dt, params, x_scale=x_scale)
    _t, _x = next_state
    return _t, _x

@partial(jit, static_argnums=(1))
def _ralston(carry, dfx, dt, params, x_scale=1.):
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
        carry: a tuple containing current time and data, i.e., (t, x)

        dfx: (ordinary) differential equation co-routine (as implemented in an
            ngc-learn component)

        dt: integration time step (also referred to as `h` in mathematics)

        params: tuple containing configuration values/hyper-parameters for the
            (ordinary) differential equation an ngc-learn component will provide

        x_scale: dampening factor to scale `x` by (Default: 1)

    Returns:
        variable values iteratively integrated/advanced to next step (`t + dt`)
    """

    t, x = carry
    dx_dt = dfx(t, x, params) ## k1
    tm, xm = _step_forward(t, x, dx_dt, dt * 0.75, x_scale)
    _dx_dt = dfx(tm, xm, params)  ## k2
    ## Note: new step is a weighted combination of k1 and k2
    summed_dx_dt = _sum_combine(dx_dt, _dx_dt, weight1=(1./3.), weight2=(2./3.))
    _t, _x = _step_forward(t, x, summed_dx_dt, dt, x_scale)
    new_carry = (_t, _x)
    return new_carry, (new_carry, carry)

@partial(jit, static_argnums=(0, 3, 4, 5, 6, 7, 8))
def solve_ode(method_name, t0, x0, T, dfx, dt, params=None, x_scale=1., sols_only=True):
    if method_name =='euler':
        method = _euler
    elif method_name == 'heun':
        method = _heun
    elif method_name == 'rk2':
        method = _rk2
    elif method_name =='rk4':
        method = _rk4
    elif method_name =='ralston':
        method = _ralston

    def scanner(carry, _):
        return method(carry, dfx, dt, params, x_scale)

    x_T, (xs_next, xs_carry) = _scan(scanner, init=(t0, x0), xs=jnp.arange(T))

    if not sols_only:
        return x_T, xs_next, xs_carry

    return xs_next


########################################################################################
########################################################################################
if __name__ == '__main__':
    
    import matplotlib.pyplot as plt
    from odes import linear_2D

    dfx = linear_2D
    x0 = jnp.array([3, -1.5])

    dt = 1e-2
    t0 = 0.
    T = 800

    (t_final, x_final), (ts_sol, sol_euler), (ts_carr, xs_carr) = solve_ode('euler', t0, x0, T=T, dfx=dfx, dt=dt, params=None, sols_only=False)
    (_, x_final), (_, sol_heun), (_, xs_carr) = solve_ode('heun', t0, x0, T=T, dfx=dfx, dt=dt, params=None, sols_only=False)
    (_, x_final), (_, sol_rk2), (_, xs_carr) = solve_ode('rk2', t0, x0, T=T, dfx=dfx, dt=dt, params=None, sols_only=False)
    (_, x_final), (_, sol_rk4), (_, xs_carr) = solve_ode('rk4', t0, x0, T=T, dfx=dfx, dt=dt, params=None, sols_only=False)
    (_, x_final), (_, sol_ralston), (_, xs_carr) = solve_ode('ralston', t0, x0, T=T, dfx=dfx, dt=dt, params=None, sols_only=False)


    plt.plot(ts_sol, sol_euler[:, 0], label='x0-Euler')
    plt.plot(ts_sol, sol_heun[:, 0], label='x0-Heun')
    plt.plot(ts_sol, sol_rk2[:, 0], label='x0-RK2')
    plt.plot(ts_sol, sol_rk4[:, 0], label='x0-RK4')
    plt.plot(ts_sol, sol_ralston[:, 0], label='x0-Ralston')

    plt.plot(ts_sol, sol_euler[:, 1], label='x1-Euler')
    plt.plot(ts_sol, sol_heun[:, 1], label='x1-Heun')
    plt.plot(ts_sol, sol_rk2[:, 1], label='x1-RK2')
    plt.plot(ts_sol, sol_rk4[:, 1], label='x1-RK4')
    plt.plot(ts_sol, sol_ralston[:, 1], label='x1-Ralston')
    plt.legend(loc='best')
    plt.grid()
    #plt.show()
    plt.savefig("integrator_plot.jpg")
