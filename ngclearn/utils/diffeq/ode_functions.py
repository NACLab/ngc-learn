import jax.numpy as jnp
import jax
from jax import jit
from functools import partial
import matplotlib.pyplot as plt

'''

x0 = jnp.array([3, -1.5])
'''
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