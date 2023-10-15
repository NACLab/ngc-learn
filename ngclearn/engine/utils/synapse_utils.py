"""
Synaptic tools and utilities.
"""
from jax import numpy as jnp, jit, vmap, random

@jit
@vmap
def compute_diagonal_norm(V):
    """
    Computes the denominator for all the projections in a single step using a vector map
    Args:
        V: the matrix to update where the dims are row x cols and each row is a vector to push apart

    Returns:
        an output matrix
    """
    return jnp.dot(V, V.T)

@jit
def compute_update(V, a=0.01, min_val=0.001, max_val=1):
    """
    Computes the update for the last vector contained in V (The last row), used in combination with pressure to
    recursively update each of the vectors in the whole collection.

    Normalizing the update to maintain the magnitude of the vector will break if clipping is used. Once the values reach
    the bound the pressure will keep trying to push past the bound and force the vector in the wrong direction.

    Args:
        V: The matrix to update where the dims are row x cols and each row is a vector to push apart

        a: the update scaler, 1 is the whole update, 0 is none of the update

        min_val: clips the end vectors to this minimum value

        max_val: clips the end vectors to this maximum value

    Returns:
        an output matrix
    """
    dia = compute_diagonal_norm(V[:, :-1].T)
    numerator = jnp.dot(V[:, :-1].T, V[:, -1])
    scales = numerator / dia
    deltas = jnp.expand_dims(jnp.dot(scales, V[:, :-1].T) * a, axis=1)
    #o_norm = jnp.linalg.norm(V[:, -1].T)
    n = jnp.clip(V[:, -1, None] - deltas, min_val, max_val)
    #n_norm = jnp.linalg.norm(n)
    return n #* (o_norm / n_norm)

@jit
def pressure(v, key, a=0.01, min_val=0.001, max_val=1):
    """
    This is the main function to call to push the collection of vectors V apart. It is non-destructive and will return a
    new collection of vectors in the same order as the input. However, it does not compute them in order from top to
    bottom so the magnitude of the pressure will be applied equally if this function is called a large number of times.

    Args:
        v: The matrix to update where the dims are row x cols and each row is a vector to push apart

        key: the PRNG key to use to compute the random order of the vectors. This needs to change with each function
            call or the order will always be the same

        a: the update scaler, 1 is the whole update, 0 is none of the update

        min_val: clips the end vectors to this minimum value

        max_val: clips the end vectors to this maximum value

    Returns:
        an output matrix
    """
    order = jnp.arange(v.shape[1])
    order = random.permutation(key, order, independent=True)
    for i in range(1, v.shape[1]):
        ui = compute_update(v[:, order[0:i+1]], a, min_val, max_val)
        v = v.at[:, order[i]].set(ui[:, 0])
    return v
