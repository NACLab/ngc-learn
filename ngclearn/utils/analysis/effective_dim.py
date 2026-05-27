from jax import numpy as jnp

def participation_ratio(Z):
    Zc = Z - Z.mean(axis=0, keepdims=True)
    cov = (Zc.T @ Zc) / (Zc.shape[0] - 1)

    tr = jnp.trace(cov)
    tr2_cov = tr * tr
    cov2_tr = jnp.trace(cov @ cov)

    return tr2_cov / cov2_tr if cov2_tr > 0 else float("nan")
