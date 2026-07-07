from jax import numpy as jnp, jit

@partial(jit, static_argnums=[1])
def participation_ratio(
    latent_codes, use_NaN_fallback=False
):
    """
    Calculates the participation ratio coefficient for a set of latent codes

    Args:
        latent_codes: a set of (N x D) latent code vectors (one row per vector code)

        use_NaN_fallback: if True, this function returns NaN for a squared covariance 
            trace of zero; else, it returns an eff-dim of 1 (Default: False)

    Returns:
        scalar measurement of the effective dimension
    """
    Z = latent_codes
    Zc = Z - Z.mean(axis=0, keepdims=True)
    cov = (Zc.T @ Zc) / (Zc.shape[0] - 1)

    tr = jnp.trace(cov)
    tr2_cov = tr * tr
    cov2_tr = jnp.trace(cov @ cov)

    ## this algorithm supports one of two fallback cases
    if not use_NaN_fallback: ## use fallback-to-1 eff-dim check
        ## use JAX-friendly conditional / direct switch to fallback to 1.0.
        ### if squared trace of covariance is 0 then effective dimension is 1.0
        return jnp.where(cov2_tr > 0.0, tr2_cov / cov2_tr, 1.0)
    ##else, use ML-oriented NaN return value fallback
    return tr2_cov / cov2_tr if cov2_tr > 0 else float("nan")

def rankme(Z, eps=1e-7):
    """
    Calculates the effective rank of for a code matrix Z effective rank = exp(Shannon entropy), adapted from:

    | Garrido, Balestriero, Najman & LeCun, "RankMe: Assessing the Downstream Performance of Pretrained
    | Self-Supervised Representations by Their Rank" (ICML 2023, arXiv:2210.02885).

    Args:
        Z: a set of (N x D) latent code vectors (one row per vector code)

        eps: (regularization) constant to prevent division by zero

    Returns:
        scalar measurement of the effective dimension
    """

    singular_values = jnp.linalg.svd(Z, compute_uv=False) ## singular values of Z
    sum_singular_vals = jnp.sum(singular_values) ## L1
    if sum_singular_vals <= 0:
        return float("nan")
    p = singular_values / sum_singular_vals + eps ## L1-normalized singular value
    shannon_entropy = -jnp.sum(p * jnp.log(p)) ## calc Shannon entropy
    return jnp.exp(shannon_entropy) ## compute final exp(Shannon entropy) = effective rank
