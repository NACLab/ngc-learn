from jax import numpy as jnp

def participation_ratio(latent_codes):
    """
    Calculates the participation ratio coefficient for a set of latent codes

    Args:
        latent_codes: a set of (N x D) latent code vectors (one row per vector code)

    Returns:
        scalar measurement of the effective dimension
    """
    Z = latent_codes
    Zc = Z - Z.mean(axis=0, keepdims=True)
    cov = (Zc.T @ Zc) / (Zc.shape[0] - 1)

    tr = jnp.trace(cov)
    tr2_cov = tr * tr
    cov2_tr = jnp.trace(cov @ cov)

    return tr2_cov / cov2_tr if cov2_tr > 0 else float("nan")

def rankme(Z, eps=1e-7):
    """
    Calculates the effective rank of for a code matrix Z effective rank = exp(Shannon entropy), adapted from:

    | Garrido, Balestriero, Najman & LeCun, "RankMe: Assessing the Downstream Performance of Pretrained
    | Self-Supervised Representations by Their Rank" (ICML 2023, arXiv:2210.02885).

    Args:
        Z: a set of (N x D) latent code vectors (one row per vector code)

    Returns:
        scalar measurement of the effective dimension
    """

    singular_values = jnp.linalg.svd(Z, compute_uv=False)   ## singular values of Z
    sum_singular_vals = jnp.sum(singular_values)            ## L1
    if sum_singular_vals <= 0:
        return float("nan")
    p = singular_values / sum_singular_vals + eps          ## L1-normalized singular value
    shannon_entropy = -jnp.sum(p * jnp.log(p))             ## Shannon entropy
    return jnp.exp(shannon_entropy)                        ## exp(Shannon entropy) = effective rank
