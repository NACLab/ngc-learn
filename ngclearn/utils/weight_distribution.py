"""
Weight distribution initialization routines and co-routines, including
parameter mapping functions for standard initializers.
"""
from jax import numpy as jnp, jit, vmap, random, lax, nn

################################################################################
## supported distribution initializer configuration generator routines

def constant(value, **kwargs):
    """
    Produce a configuration for a constant weight distribution initializer.

    Args:
        value: magnitude of the weight values (shared across all)

    Returns:
        a constant weight initializer configuration
    """
    dist_dict = {"dist": "constant", "value": value}
    return {**kwargs, **dist_dict}

def fan_in_gaussian(**kwargs):
    """
    Produce a configuration for a fan-in scaled (centered) Gaussian
    distribution initializer.

    Returns:
        a fan-in scaled Gaussian distribution configuration
    """
    dist_dict = {"dist": "fan_in_gaussian"}
    return {**kwargs, **dist_dict}

def gaussian(mu=0., sigma=1., **kwargs):
    """
    Produce a configuration for a Gaussian distribution initializer.

    Args:
        mu: mean of the weight values (default: 0)

        sigma: standard deviation of the weight values (default: 1)

    Returns:
        a Gaussian distribution configuration
    """
    assert sigma >= 0.
    dist_dict = {"dist": "gaussian", "mu": mu, "sigma": sigma}
    return {**kwargs, **dist_dict}

def uniform(amin=0., amax=1., **kwargs):
    """
    Produce a configuration for a uniform distribution initializer.

    Args:
        amin: minimum value/bound of weight values (default: 0)

        amax: maximum value/bound of weight values  (default: 1)

    Returns:
        a uniform distribution configuration
    """
    assert amin < amax
    dist_dict = {"dist": "uniform", "amin": amin, "amax": amax}
    return {**kwargs, **dist_dict}

def hollow(scale, **kwargs):
    """
    Produce a configuration for a constant hollow distribution initializer.

    Args:
        scale: magnitude of all off-diagonal values

    Returns:
        a constant hollow distribution configuration
    """
    dist_dict = {"dist": "hollow", "scale": scale}
    return {**kwargs, **dist_dict}

def eye(scale, **kwargs):
    """
    Produce a configuration for a constant diagonal/eye distribution initializer.

    Args:
        scale: magnitude of all (on-)diagonal values

    Returns:
        a constant diagonal/eye distribution configuration
    """
    dist_dict = {"dist": "eye", "scale": scale}
    return {**kwargs, **dist_dict}

################################################################################
## initializer co-routine(s)

def initialize_params(dkey, init_kernel, shape):
    """
    Creates the intiial condition values for a parameter tensor.

    Args:
        dkey: PRNG key to control determinism of this routine

        init_kernel: dictionary specifying the distribution type and its
            parameters (default: `uniform` dist w/ `amin=0.02`, `amax=0.8`)

            :Note: Currently supported distribution (dist) kernel schemes include:
                "constant" (value);
                "uniform" (amin, amax);
                "gaussian" (mu, sigma);
                "fan_in_gaussian" (NO params);
                "hollow" (scale);
                "eye" (scale);
                while currently supported post-processing keyword arguments include:
                "amin" (clip weights values to be >= amin);
                "amax" (clip weights values to be <= amin);
                "hollow" (zero out values along main diagonal);
                "eye" (zero out off-diagonal values)

        shape: tuple containing the dimensions/shape of the tensor to initialize

    Returns:
        output (tensor) value
    """
    _init_kernel = init_kernel
    if _init_kernel is None: ## the "universal default distribution" if None provided
        _init_kernel = {"dist": "uniform", "amin": 0.025, "amax": 0.8}
    dist_type = _init_kernel.get("dist")
    params = None
    if dist_type == "hollow":
        diag_scale = _init_kernel.get("scale", 1.)
        params = (1. - jnp.eye(N=shape[0], M=shape[1])) * diag_scale
    elif dist_type == "eye":
        off_diag_scale = _init_kernel.get("scale", 1.)
        params = jnp.eye(N=shape[0], M=shape[1]) * off_diag_scale
    elif dist_type == "gaussian" or dist_type == "normal":
        mu = _init_kernel.get("mu", 0.)
        sigma = _init_kernel.get("sigma", 1.)
        params = random.normal(dkey, shape) * sigma + mu
    elif dist_type == "uniform":
        amin = _init_kernel.get("amin", 0.)
        amax = _init_kernel.get("amax", 1.)
        params = random.uniform(dkey, shape, minval=amin, maxval=amax)
    elif dist_type == "fan_in_gaussian":
        phi = random.normal(dkey, shape)
        phi = phi * jnp.sqrt(1.0 / (shape[0] * 1.))
        params = phi.astype(jnp.float32)
    elif dist_type == "constant":
        scale = _init_kernel.get("value", 1.)
        params = jnp.ones(shape) * scale
    else:
        raise RuntimeError(
            "Initialization scheme (" + dist_type + ") is not recognized/supported!"
        )
    ## check for any additional distribution post-processing kwargs (e.g., clipping)
    clip_min = _init_kernel.get("amin")
    clip_max = _init_kernel.get("amax")
    is_hollow = _init_kernel.get("hollow", False)
    is_eye = _init_kernel.get("eye", False)
    if clip_min is not None: ## bound all values to be > clip_min
        params = jnp.maximum(params, clip_min)
    if clip_max is not None: ## bound all values to be < clip_max
        params = jnp.minimum(params, clip_max)
    if is_hollow: ## apply a hollow mask
        params = (1. - jnp.eye(N=shape[0], M=shape[1])) * params
    if is_eye: ## apply an eye/diagonal mask
        params = jnp.eye(N=shape[0], M=shape[1]) * params
    return params ## return initial distribution conditions

