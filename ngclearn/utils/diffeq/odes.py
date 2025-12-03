"""
In-built dynamical systems built on differential equations. Note that these systems are designed such that they
directly operzte with ngc-learn's ODE integration backend.

| Currently in-built dynamical systems include:
| 0) A continuous linear 2D system;
| 1) A continuous cubic 2D system;
| 2) A Lorenz attractor system;
| 3) A continuous linear 3D system;
| 4) A continuous oscillator system.

"""

import jax.numpy as jnp

def linear_2D(t, x, params):
    """
    * suggested init value - x0 = jnp.array([3, -1.5])

    Args:
        x: 2D vector
               type: jax array
               shape:(2,)

        t: Unused

        params: Unused

    Returns:
        2D vector:
            [
            -0.1 * x[0] + 2.0 * x[1],
            -2.0 * x[0] - 0.1 * x[1]
            ]; type: jax array, shape:(2,)
    """
    coeff = jnp.array([[-0.1, 2],
                       [-2, -0.1]]).T
    dfx_ = jnp.matmul(x, coeff)

    return dfx_

def cubic_2D(t, x, params):
    """
    suggested init value - x0 = jnp.array([2., 0.])

    Args:
        x: 2D vector
                  type: jax array
                  shape: (2,)

        t: Unused

        params: Unused

    Returns:
        2D vector:
            [
            -0.1 * x[0] ** 3 + 2.0 * x[1] ** 3,
            -2.0 * x[0] ** 3 - 0.1 * x[1] ** 3,
            ]; type: jax array, shape:(2,)
    """
    coeff = jnp.array([[-0.1, 2],
                       [-2, -0.1]]).T
    dfx_ = jnp.matmul(x**3, coeff)
    return dfx_

def lorenz(t, x, params):
    """
    suggested init value - x0 = jnp.array([-8, 7, 27])

    Args:
        x: 3D vector
                  type: jax array
                  shape: (3,)

        t: Unused

        params: Unused

    Returns:
        3D vector:
            [
            10 * (x[1] - x[0]),
            x[0] * (28 - x[2]) - x[1],
            x[0] * x[1] - 8 / 3 * x[2],
            ]; type: jax array, shape:(3,)
    """
    x_ = x[..., 0]
    y_ = x[..., 1]
    z_ = x[..., 2]

    dx = 10 * y_ - 10 * x_
    dy = 28 * x_ - x_ * z_ - y_
    dz = x_ * y_ - 8 / 3 * z_
    return jnp.stack([dx, dy, dz], axis=-1)


def linear_3D(t, x, params):
    """
    suggested init value - x0 = jnp.array([1, 1., -1])

    Args:
        x: 3D vector
                  type: jax array
                  shape: (3,)

        t: Unused

        params: Unused

    Returns:
        3D vector:
            [
            -0.1 * x[0] + 2 * x[1],
            -2 * x[0] - 0.1 * x[1],
            -0.3 * x[2]
            ]; type: jax array, shape:(3,)
    """
    x_ = x[..., 0]
    y_ = x[..., 1]
    z_ = x[..., 2]

    dx = -0.1 * x_ + 2.0 * y_
    dy =  -2.0 * x_ - 0.1 * y_
    dz = -0.3 * z_

    return jnp.stack([dx, dy, dz], axis=-1)

def oscillator(t, x, params, mu1=0.05, mu2=-0.01, omega=3.0, alpha=-2.0, beta=-5.0, sigma=1.1):
    """
    suggested init value - x0 = jnp.array([0.5, 0.05, 0.1])

    Args:
        x: 3D vector
                  type: jax array
                  shape: (3,)

        t: Unused

        params: Unused

    Returns:
        3D vector:
            [
            mu1 * x[0] + sigma * x[0] * x[1],
            mu2 * x[1] + (omega + alpha * x[1] + beta * x[2]) * x[2] - sigma * x[0] ** 2,
            mu2 * x[2] - (omega + alpha * x[1] + beta * x[2]) * x[1],
            ]; type: jax array, shape:(3,)
    """
    x_ = x[..., 0]
    y_ = x[..., 1]
    z_ = x[..., 2]

    dx = mu1 * x_ + sigma * x_ * y_
    dy = mu2 * y_ + (omega + alpha * y_ + beta * z_) * z_ - sigma * x_ ** 2
    dz = mu2 *z_ - omega *y_ - alpha * y_*y_ - beta * z_* y_

    return jnp.stack([dx, dy, dz], axis=-1)


## some testing/driver code to check the ODEs themselves
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from ngclearn.utils.diffeq.ode_utils import solve_ode


    t0 = 0.
    dt = 0.01

    # 1. Linear 2D System
    x0 = jnp.array([3, -1.5], dtype=jnp.float32)
    t, x_lin2D = solve_ode('rk4', t0=t0, x0=x0, T=3000, dfx=linear_2D, dt=dt, params=None, sols_only=True)
    # 2. Cubic 2D System
    x0 = jnp.array([2, 0.], dtype=jnp.float32)
    t, x_cub2D = solve_ode('rk4', t0=t0, x0=x0, T=10000, dfx=cubic_2D, dt=dt, params=None, sols_only=True)
    # 3. Lorenz System (3D)
    x0 = jnp.array([-8, 7, 27], dtype=jnp.float32)
    t, x_lorenz = solve_ode('rk4', t0=t0, x0=x0, T=2000, dfx=lorenz, dt=dt, params=None, sols_only=True)
    # 4. Linear 3D System
    x0 = jnp.array([1, 1., -1], dtype=jnp.float32)
    t, x_lin3D = solve_ode('rk4', t0=t0, x0=x0, T=10000, dfx=linear_3D, dt=dt, params=None, sols_only=True)
    # 5. Oscillator System
    x0 = jnp.array([0.5, 0.05, 0.1], dtype=jnp.float32)
    t, x_osci = solve_ode('rk4', t0=t0, x0=x0, T=20000, dfx=oscillator, dt=dt, params=None, sols_only=True)


    plt.plot(x_lin2D[:, 0], x_lin2D[:, 1], linewidth=2, color='darkorange', label=r'$linear-2D$')
    plt.title('Linear 2D System', fontsize=20)
    plt.xlabel('x', fontsize=20)
    plt.ylabel('y', fontsize=20)
    plt.grid(True)
    plt.show()

    plt.plot(x_cub2D[:, 0], x_cub2D[:, 1], linewidth=2, color='royalblue', label=r'cubic-2D$')
    plt.title('Cubic 2D System', fontsize=20)
    plt.xlabel('x', fontsize=20)
    plt.ylabel('y', fontsize=20)
    plt.grid(True)
    plt.show()

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot(x_lorenz[:, 0], x_lorenz[:, 1], x_lorenz[:, 2], linewidth=1, color='red', label=r'$lorenz$')
    ax.set_title('Lorenz System', fontsize=20)
    ax.set_xlabel('x', fontsize=20)
    ax.set_ylabel('y', fontsize=20)
    ax.set_zlabel('z', fontsize=20)
    plt.grid(True)
    plt.show()


    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot(x_lin3D[:, 0], x_lin3D[:, 1], x_lin3D[:, 2], linewidth=1, color='purple', label=r'linear-3D')
    ax.set_title('Linear 3D System', fontsize=20)
    ax.set_xlabel('x', fontsize=20)
    ax.set_ylabel('y', fontsize=20)
    ax.set_zlabel('z', fontsize=20)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot(x_osci[:, 0], x_osci[:, 1], x_osci[:, 2], linewidth=1, color='green', label=r'oscillator')
    ax.set_title('Atmospheric Oscillator', fontsize=20)
    ax.set_xlabel('x', fontsize=20)
    ax.set_ylabel('y', fontsize=20)
    ax.set_zlabel('z', fontsize=20)
    plt.grid(True)
    plt.show()







