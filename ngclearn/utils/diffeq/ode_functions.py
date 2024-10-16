import jax.numpy as jnp
import jax
from jax import jit
from functools import partial
import matplotlib.pyplot as plt


@partial(jit, static_argnums=(0,))
def linear_2D(t, x, params):
    '''
    :param x: 2D vector
            type: jax array
            shape:(2,)

    :param t: Unused

    :param params: Unused

    :return: 2D vector: [
                         -0.1 * x[0] + 2.0 * x[1],
                         -2.0 * x[0] - 0.1 * x[1]
                         ]
            type: jax array
            shape:(2,)

    ------------------------------------------
        * suggested init value-
                x0 = jnp.array([3, -1.5])
    '''
    coeff = jnp.array([[-0.1, 2], [-2, -0.1]]).T
    dfx_ = jnp.matmul(x, coeff)

    return dfx_



class CreateLibrary:
    def __init__(self):
        pass

    @staticmethod
    def poly_2D(x, y, deg=2, include_bias=True):
        x = jnp.array(x).reshape(-1, 1)
        y = jnp.array(y).reshape(-1, 1)
        lib = jnp.ones_like(x).reshape(-1, 1)
        names = ['1']
        for i in range(deg + 1):
            for j in range(deg - i + 1):
                lib = jnp.concatenate([lib, x ** i * y ** j], axis=1)
                names.append('x^{} y^{}'.format(i, j))
        if include_bias:
            return lib[:, 1:], names
        else:
            return lib[:, 2:], names[1:]
