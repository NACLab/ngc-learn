"""
Weight distribution initialization routines and co-routines, including
parameter mapping functions for standard initializers.
"""
import numpy as np
import jax
from jax import numpy as jnp, jit, vmap, lax, nn, random
from ngcsimlib.logger import critical

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

def fan_in_uniform(**kwargs):
    """
    Produce a configuration for a fan-in scaled unit uniform
    distribution initializer.

    Returns:
        a fan-in scaled (unit) uniform distribution configuration
    """
    dist_dict = {"dist": "fan_in_uniform"}
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

def initialize_params(dkey, init_kernel, shape, use_numpy=False):
    """
    Creates the intiial condition values for a parameter tensor.

    Args:
        dkey: PRNG key to control determinism of this routine

        init_kernel: dictionary specifying the distribution type and its
            parameters (default: `uniform` dist w/ `amin=0.02`, `amax=0.8`) --
            note that kernel dictionary may contain "post-processing" arguments
            that can be "stacked" on top of the base matrix, for example, you
            can pass in a dictionary:
            {"dist": "uniform", "hollow": True, "lower_triangle": True} which
            will create unit-uniform value matrix with upper triangle and main
            diagonal values masked to zero (lower-triangle masking applied after
            hollow matrix masking)

            :Note: Currently supported distribution (dist) kernel schemes include:
                "constant" (value);
                "uniform" (amin, amax);
                "gaussian" (mu, sigma);
                "fan_in_gaussian" (NO params);
                "fan_in_uniform" (NO params);
                "hollow" (scale);
                "eye" (scale);
                while currently supported post-processing keyword arguments include:
                "amin" (clip weights values to be >= amin);
                "amax" (clip weights values to be <= amin);
                "lower_triangle" (extract lower triangle of params, set rest to 0);
                "upper_triangle" (extract upper triangle of params, set rest to 0);
                "hollow" (zero out values along main diagonal);
                "eye" (zero out off-diagonal values);
                "n_row_active" (keep only n random rows non-masked/zero);
                "n_col_active" (keep only n random columns non-masked/zero)

        shape: tuple containing the dimensions/shape of the tensor to initialize

        use_numpy: if true, conducts weight value initialization/post-processing using
            exclusively Numpy, disabling Jax calls (default: False)

    Returns:
        output (tensor) value
    """
    if dkey is None:
        use_numpy = True

    _init_kernel = init_kernel
    if _init_kernel is None: ## the "universal default distribution" if None provided
        critical("No initialization kernel provided!")
    dist_type = _init_kernel.get("dist")
    params = None
    if dist_type == "hollow": ## scaled hollow-matrix init
        diag_scale = _init_kernel.get("scale", 1.)
        if use_numpy:
            params = (1. - np.eye(N=shape[0], M=shape[1])) * diag_scale
        else:
            params = (1. - jnp.eye(N=shape[0], M=shape[1])) * diag_scale
    elif dist_type == "eye": ## scaled diagonal/eye init
        off_diag_scale = _init_kernel.get("scale", 1.)
        if use_numpy:
            params = np.eye(N=shape[0], M=shape[1]) * off_diag_scale
        else:
            params = jnp.eye(N=shape[0], M=shape[1]) * off_diag_scale
    elif dist_type == "gaussian" or dist_type == "normal": ## normal distrib
        mu = _init_kernel.get("mu", 0.)
        sigma = _init_kernel.get("sigma", 1.)
        if use_numpy:
            params = np.random.normal(size=shape) * sigma + mu
        else:
            params = jax.random.normal(dkey, shape) * sigma + mu
    elif dist_type == "uniform": ## uniform distrib
        amin = _init_kernel.get("amin", 0.)
        amax = _init_kernel.get("amax", 1.)
        if use_numpy:
            params = np.random.uniform(low=amin, high=amax, size=shape)
        else:
            params = jax.random.uniform(dkey, shape, minval=amin, maxval=amax)
    elif dist_type == "fan_in_gaussian": ## fan-in scaled standard normal init
        if use_numpy:
            phi = np.random.normal(size=shape)
        else:
            phi = jax.random.normal(dkey, shape)
        phi = phi * jnp.sqrt(1.0 / (shape[0] * 1.))
        params = phi.astype(jnp.float32)
    elif dist_type == "fan_in_uniform": ## fan-in scaled unit uniform init
        phi = jnp.sqrt(1.0 / (shape[0] * 1.)) # sometimes "k" in other libraries
        if use_numpy:
            params = np.random.uniform(low=-phi, high=phi, size=shape)
        else:
            params = jax.random.uniform(dkey, shape, minval=-phi, maxval=phi)
        params = params.astype(jnp.float32)
    elif dist_type == "constant": ## constant value (everywhere) init
        scale = _init_kernel.get("value", 1.)
        if use_numpy:
            params = np.ones(shape) * scale
        else:
            params = jnp.ones(shape) * scale
    else:
        critical("Initialization scheme (" + dist_type + ") is not recognized/supported!")
    ## check for any additional distribution post-processing kwargs (e.g., clipping)
    clip_min = _init_kernel.get("amin")
    clip_max = _init_kernel.get("amax")
    lower_triangle = init_kernel.get("lower_triangle", False)
    upper_triangle = init_kernel.get("upper_triangle", False)
    is_hollow = _init_kernel.get("hollow", False)
    is_eye = _init_kernel.get("eye", False)
    n_row_active = _init_kernel.get("n_row_active", None)
    n_col_active = _init_kernel.get("n_col_active", None)
    block_diag_mask_width = _init_kernel.get("block_diag_mask_width", None)
    ## run any configured post-processing to condition the final value matrix
    if clip_min is not None: ## bound all values to be > clip_min
        if use_numpy:
            params = np.maximum(params, clip_min)
        else:
            params = jnp.maximum(params, clip_min)
    if clip_max is not None: ## bound all values to be < clip_max
        if use_numpy:
            params = np.minimum(params, clip_max)
        else:
            params = jnp.minimum(params, clip_max)
    if block_diag_mask_width is not None:
        k = int(params.shape[0] / block_diag_mask_width) #5
        n = block_diag_mask_width #2
        source = jnp.eye(k, k)
        block_mask = jnp.repeat(jnp.repeat(source, n, axis=1), n, axis=0)
        if block_mask.shape[0] == params.shape[0] and block_mask.shape[1] == params.shape[1]:
            params = params * block_mask
        else:
            critical(
                "Initialization block matrix w/ width (" + block_diag_mask_width +
                ") is not recognized/supported!"
            )
    if lower_triangle: ## extract lower triangle of params matrix
        ltri_params = jax.numpy.tril(params.shape[0])
        params = ltri_params
    if upper_triangle: ## extract upper triangle of params matrix
        ltri_params = jax.numpy.triu(params.shape[0])
        params = ltri_params
    if is_hollow: ## apply a hollow mask
        if use_numpy:
            params = (1. - np.eye(N=shape[0], M=shape[1])) * params
        else:
            params = (1. - jnp.eye(N=shape[0], M=shape[1])) * params
    if is_eye: ## apply an eye/diagonal mask
        if use_numpy:
            params = np.eye(N=shape[0], M=shape[1]) * params
        else:
            params = jnp.eye(N=shape[0], M=shape[1]) * params
    if n_row_active is not None:  ## keep only n rows active (rest are zero)
        row_ind = random.permutation(dkey, shape[0])[0:n_row_active]
        mask = jnp.zeros(shape)
        mask = mask.at[row_ind, :].set(jnp.ones((shape[0], 1))) ## only set keep rows to ones
        params = params * mask
    if n_col_active is not None:  ## keep only n cols active (rest are zero)
        row_col = random.permutation(dkey, shape[1])[0:n_col_active]
        mask = jnp.zeros(shape)
        mask = mask.at[:, row_col].set(jnp.zeros((1, shape[0]))) ## only set keep cols to ones
        params = params * mask

    return params ## return initial distribution conditions

