from functools import partial
import jax
from jax import numpy as jnp, jit

@partial(jit, static_argnums=[1])
def participation_ratio(
    latent_codes, use_NaN_fallback=False
):
    """
    Calculates the participation ratio coefficient (also known as the Gini effective 
    dimension) for a set of latent codes. 

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
    ## calc frob-norm squared; NOTE: faster line replaced older un-commented one to right
    cov2_tr = jnp.sum(jnp.square(cov)) #cov2_tr = jnp.trace(cov @ cov)
    
    ## this algorithm supports one of two fallback cases
    if not use_NaN_fallback: ## use fallback-to-1 eff-dim check
        ## use JAX-friendly conditional / direct switch to fallback to 1.0.
        ### if squared trace of covariance is 0 then effective dimension is 1.0
        return jnp.where(cov2_tr > 0.0, tr2_cov / cov2_tr, 1.0)
    ##else, use ML-oriented NaN return value fallback
    return tr2_cov / cov2_tr if cov2_tr > 0 else float("nan")



@partial(jit, static_argnums=[1])
def rankme(latent_codes, eps=1e-7):
    """
    Calculates the effective rank of for a code matrix latent_codes

    effective rank = exp(Shannon entropy), adapted from:
    | Garrido, Balestriero, Najman & LeCun, "RankMe: Assessing the Downstream Performance of Pretrained
    | Self-Supervised Representations by Their Rank" (ICML 2023, arXiv:2210.02885).

    Args:
        latent_codes: a set of (N x D) latent code vectors (one row per vector code)

        eps: (regularization) constant to prevent division by zero

    Returns:
        scalar measurement of the effective dimension
    """

    singular_values = jnp.linalg.svd(latent_codes, compute_uv=False) ## singular values of latent_codes
    sum_singular_values = jnp.sum(singular_values)                   ## L1
    sum_S_vals = jnp.where(sum_singular_values > 0.0, sum_singular_values, 1.0)
    p = singular_values / (sum_S_vals + eps)                         ## L1-normalized singular value
    safe_p = jnp.where(p > 0.0, p, 1.0)
    shannon_entropy = -jnp.sum(p * jnp.log(safe_p))                       ## calc Shannon entropy

    ## compute final exp(Shannon entropy) = effective rank
    #rankme_score = jnp.exp( ## compute final exp(Shannon entropy) = effective rank
    #    jnp.where(sum_singular_values > 0.0, shannon_entropy, jnp.nan)
    #)
    rankme_score = jnp.where(sum_singular_values > 0.0, jnp.exp(shannon_entropy), 1.0)
    return rankme_score

@partial(jit, static_argnums=[1])
def stable_rank(latent_codes, num_iters=10): ## power-iterator method
    """
    Computes the stable rank via the power iteration method in order to find the 
    top singular value.

    Args:
        latent_codes: a set of (N x D) latent code vectors (one row per vector code)

        num_iters: number of iterations to run power iterator calculation (Default: 10)

    Returns:
        scalar measurement of the stable rank (proxy for effective dimension)
    """
    Z = latent_codes
    Zc = Z - Z.mean(axis=0, keepdims=True) ## center codes
    frobenius_norm_sq = jnp.sum(jnp.square(Zc)) ## calc squared frob-norm
    
    ## use power iteration to find dominant eigenvector of Zc.T @ Zc
    ### start w/ random vector: 
    key = jax.random.PRNGKey(0)
    v = jax.random.normal(key, (Zc.shape[1], 1))
    v = v / jnp.linalg.norm(v)
    ## apply standard power iteration loop
    for _ in range(num_iters):
        ## v = (Zc.T @ (Zc @ v))
        v = Zc.T @ (Zc @ v)
        v = v / jnp.linalg.norm(v)
    ## compute largest singular value squared (i.e., the Rayleigh quotient): 
    ### sigma_max^2 = ||Zc @ v||^2
    sigma_max_sq = jnp.sum(jnp.square(Zc @ v)) 
    return jnp.where(sigma_max_sq > 0.0, frobenius_norm_sq / sigma_max_sq, 1.0) # stable-rank score


