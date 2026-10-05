from functools import partial
import jax
from jax import numpy as jnp, jit
from ngcsimlib import deprecated

'''
This sub-module contains measurement functions for analyzing properties of vector codes, particularly for the case 
of dimensional analysis (to investigate issues surrounding representational/dimensional collapse and dimensional 
usage/capacity). Some routines provided (e.g., `measure_excecss_kurtosis(.)`) come from the independent component analysis 
(ICA) literature and help investigate if vector codes lean towards ICA-like patterns. 

Some useful notes on effective dimensional analysis: 

* Participation ratio (PR), which measures the general usage of a vector space (how many 
features are being used), can be easily "fooled" by a "bully" dimension, specifically yielding 
cases where, say all D dimensions are all active but one of them holds 99% of variance while 
the other D-1 dims share the remaining 1%; in this case, PR would yield a rather high, seemingly 
healthy-looking score yet it is not accounting for the fact that other low-variance dims are 
participating yet are far too "quiet"

* Stable rank (SR; which is a function of the Rayleigh coefficient) is good with detecting 
if a single feature/dimension is 
completely drowning out the rest of the vector space - if stable rank goes close to 1, then 
model has collapsed to a 1-dim case even though the PR is high; this metric is useful to 
examine to check if a vector space is multi-dimensional and balanced (and not just a single 
massive eigenvector surrounded by insignificant/low-contributing dimensions)

PR, SR, and Rankme are metrics along a spectral analysis metric spectrum: 
* Rankme is the exponential Shannon entropy of spectrum, 
* PR is the Renyi-2 "effective dimension", and, 
* SR focuses on the single largest eigenvalue of the dimensional space
'''

@partial(jit, static_argnums=[1])
def measure_participation_ratio(latent_codes, use_NaN_fallback=False):
    """
    Calculates the participation ratio (PR) coefficient (also known as the Gini effective 
    dimension) for a set of latent codes. PR is useful for detecting "total dimensional 
    collapse", where the data/vector-space essentially flattens into a line (or only 
    make use of too few or even just a single dimension of the space).

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

@deprecated(replaced_by=measure_participation_ratio)
def participation_ratio(latent_codes, use_NaN_fallback=False):
    """
    WARNING: this function is deprecated (renamed to `measure_participation_ratio(.)`).
    """
    return measure_participation_ratio(latent_codes, use_NaN_fallback=use_NaN_fallback)

@jit
def measure_covariance_error(latent_codes):
    """
    Calculates the off-diagonal covariance error of a set of latent codes. This dimensional metric is useful for
    quantifying informational redundancy. If the error/score is high, units/dimensions are highly correlated, which
    means the vector code space is wasting its dimensional capacity by having different dimensions/features model
    the exact same piece of information.

    Args:
        latent_codes: a set of (N x D) latent code vectors (one row per vector code)

    Returns:
        scalar measurement of the off-diagonal covariance error
    """
    Z = latent_codes
    Zc = Z - Z.mean(axis=0, keepdims=True)
    cov = (Zc.T @ Zc) / (Zc.shape[0] - 1)
    ## normalize covariance to get correlation matrix
    d = jnp.diag(cov)
    std_dev = jnp.sqrt(jnp.clip(d, a_min=1e-8))
    denominator = std_dev[:, None] * std_dev[None, :]
    corr = cov / jnp.clip(denominator, a_min=1e-6)
    ## zero out diagonal elements
    # diag_mask = jnp.eye(corr.shape[0])
    # off_diag = corr * (1.0 - diag_mask)
    ## calc mean squared off-diagonal error
    # off_diagonal_error = jnp.sum(off_diag ** 2) / (corr.shape[0] * (corr.shape[0] - 1))
    off_diagonal_error = (jnp.sum(corr ** 2) - jnp.sum(jnp.diag(corr) ** 2)) / (corr.shape[0] * (corr.shape[0] - 1))
    return off_diagonal_error

@deprecated(replaced_by=measure_covariance_error)
def covariance_error(latent_codes):
    """
    WARNING: this function is deprecated (renamed to `measure_covariance_error(.)`).
    """
    return measure_covariance_error(latent_codes)

@jit
def measure_excess_kurtosis(y):
    """
    Computes the sample excess kurtosis for each neuron/unit across a batch; this then takes the mean over all neurons/units.

    Args:
        y: Latent activations batch matrix of shape (B, H)

    Returns:
        mean_kurtosis: scalar value (>0 indicates super-Gaussianity)
    """
    ## center the hidden activities relative to the batch mean
    y_centered = y - jnp.mean(y, axis=0, keepdims=True)

    ## compute the 2nd and 4th central moments
    variance = jnp.mean(y_centered ** 2, axis=0) + 1e-8
    fourth_moment = jnp.mean(y_centered ** 4, axis=0)

    ## calculate excess kurtosis (Pearson's kurtosis minus 3)
    neuron_kurtosis = (fourth_moment / (variance ** 2)) - 3.0

    ## Return the average sparsity signature across the entire layer profile
    return jnp.mean(neuron_kurtosis)

@partial(jit, static_argnums=[1])
def measure_stable_rank(latent_codes, num_iters=10): ## built on power-iterator method
    """
    Computes the "stable rank} via the power iteration method in order to find the 
    top singular value (this metric is a function of the Rayleigh coefficient). Note that 
    this metric is useful for detecting a case of dimensional collapse known as "dominant 
    component collapse", where a single feature "hogs" up all 
    the power of the representational vector space while ignoring everything else. 

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
    ## run power iteration loop
    for _ in range(num_iters):
        ## v = (Zc.T @ (Zc @ v))
        v = Zc.T @ (Zc @ v)
        v = v / jnp.linalg.norm(v)
    ## compute largest singular value squared => sigma_max^2 = ||Zc @ v||^2
    sigma_max_sq = jnp.sum(jnp.square(Zc @ v)) ## Rayleigh coefficient/quotient
    return jnp.where(sigma_max_sq > 0.0, frobenius_norm_sq / sigma_max_sq, 1.0) # stable-rank score

@deprecated(replaced_by=measure_stable_rank)
def stable_rank(latent_codes, num_iters=10):
    """
    WARNING: this function is deprecated (renamed to `measure_stable_rank(.)`).
    """
    return measure_stable_rank(latent_codes, num_iters=num_iters)

@partial(jit, static_argnums=[1])
def measure_rankme(latent_codes, eps=1e-7):
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

@deprecated(replaced_by=measure_rankme)
def rankme(latent_codes, eps=1e-7):
    """
    WARNING: this function is deprecated (renamed to `measure_rankme(.)`).
    """
    return measure_rankme(latent_codes, eps=eps)
