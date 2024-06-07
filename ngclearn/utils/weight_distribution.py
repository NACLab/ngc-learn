"""
Weight distribution initialization routines and co-routines, including
parameter mapping functions for standard initializers.
"""
from jax import numpy as jnp, jit, vmap, random, lax, nn

################################################################################
## supported distribution initializer configuration generator routines

def constant(value):
    """
    Produce a configuration for a constant weight distribution initializer.

    Args:
        value: magnitude of the weight values (shared across all)

    Returns:
        a constant weight initializer configuration
    """
    return "constant", value, 0.

def fan_in_gaussian():
    """
    Produce a configuration for a fan-in scaled (centered) Gaussian
    distribution initializer.

    Returns:
        a fan-in scaled Gaussian distribution configuration
    """
    return "fan_in_gaussian", 0., 0.

def gaussian(mu=0., sigma=1.):
    """
    Produce a configuration for a Gaussian distribution initializer.

    Args:
        mu: mean of the weight values (default: 0)

        sigma: standard deviation of the weight values (default: 1)

    Returns:
        a Gaussian distribution configuration
    """
    return "gaussian", mu, sigma

def uniform(minval=0., maxval=1.):
    """
    Produce a configuration for a uniform distribution initializer.

    Args:
        minval: minimum value/bound of weight values (default: 0)

        maxval: maximum value/bound of weight values  (default: 1)

    Returns:
        a uniform distribution configuration
    """
    return "uniform", minval, maxval

def hollow(scale):
    """
    Produce a configuration for a constant hollow distribution initializer.

    Args:
        scale: magnitude of all off-diagonal values

    Returns:
        a constant hollow distribution configuration
    """
    return "hollow", scale, 0.

def hollow_uniform(minval=0., maxval=1.):
    """
    Produce a configuration for a hollow-masked uniform distribution initializer.

    Args:
        minval: minimum value/bound of off-diagonal weight values (default: 0)

        maxval: maximum value/bound of off-diagonal weight values (default: 1)

    Returns:
        a hollow-masked uniform distribution configuration
    """
    return "hollow_uniform", minval, maxval

def hollow_gaussian(mu=0., sigma=1.):
    """
    Produce a configuration for a hollow-masked Gaussian distribution initializer.

    Args:
        mu: mean of off-diagonal weight values (default: 0)

        sigma: standard deviation of off-diagonal weight values (default: 1)

    Returns:
        a hollow-masked gaussian distribution configuration
    """
    return "hollow_gaussian", mu, sigma

def eye(scale):
    """
    Produce a configuration for a constant diagonal/eye distribution initializer.

    Args:
        scale: magnitude of all (on-)diagonal values

    Returns:
        a constant diagonal/eye distribution configuration
    """
    return "eye", scale, 0.

def eye_uniform(minval=0., maxval=1.):
    """
    Produce a configuration for a diagonal/eye-masked uniform distribution initializer.

    Args:
        minval: minimum value/bound of (on-)diagonal weight values (default: 0)

        maxval: maximum value/bound of (on-)diagonal weight values (default: 1)

    Returns:
        a digonal/eye-masked uniform distribution configuration
    """
    return "eye_uniform", minval, maxval

def eye_gaussian(mu=0., sigma=1.):
    """
    Produce a configuration for a diagonal/eye-masked Gaussian distribution initializer.

    Args:
        mu: mean of (on-)diagonal weight values (default: 0)

        sigma: standard deviation of (on-)diagonal weight values (default: 1)

    Returns:
        a diagonal/eye-masked gaussian distribution configuration
    """
    return "eye_gaussian", mu, sigma

################################################################################
## initializer co-routine(s)

def initialize_params(dkey, init_kernel, shape):
    """
    Creates the intiial condition values for a parameter tensor.

    Args:
        dkey: PRNG key to control determinism of this routine

        init_kernel: triplet/3-tuple with 1st element as a string calling the name
            of initialization scheme to use

            :Note: Currently supported kernel schemes include:
                ("constant", magnitude, ~ignored~);
                ("uniform", min_val, max_val);
                ("fan_in_gaussian", ~ignored, ~ignored~);
                ("gaussian", mu, sigma) OR ("normal", mu, sigma);
                ("hollow", off_diagonal_scale, ~ignored~);
                ("hollow_uniform", off_diagonal_min_val, off_diagonal_max_val);
                ("hollow_gaussian", off_diagonal_mu, off_diagonal_sigma);
                ("eye", diagonal_scale, ~ignored~);
                ("eye_uniform", diagonal_min_val, diagonal_max_val);
                ("eye_gaussian", diagonal_mu, diagonal_sigma)

        shape: tuple containing the dimensions/shape of the tensor to initialize

    Returns:
        output (tensor) value
    """
    initType, *args = init_kernel # get out arguments of initialization kernel
    params = None
    if "hollow" in initType: ## hollow-matrix init types
        if initType == "hollow_uniform": ## hollow-uniform
            lb, ub = args
            eps = random.uniform(dkey, shape, minval=lb, maxval=ub)
            params = (1. - jnp.eye(N=shape[0], M=shape[1])) * eps
        elif initType == "hollow_gaussian":  ## hollow-normal
            mu, sigma = args
            eps = random.uniform(dkey, shape, minval=mu, maxval=sigma)
            params = (1. - jnp.eye(N=shape[0], M=shape[1])) * eps
        else: ## constant hollow
            diagScale, _ = args
            params = (1. - jnp.eye(N=shape[0], M=shape[1])) * diagScale
    elif "eye" in initType:  ## diagonal eye matrix init types
        if initType == "eye_uniform": ## diagonal-uniform
            lb, ub = args
            eps = random.uniform(dkey, shape, minval=lb, maxval=ub)
            params = jnp.eye(N=shape[0], M=shape[1]) * eps
        elif initType == "eye_gaussian": ## diagonal-normal
            mu, sigma = args
            eps = random.uniform(dkey, shape, minval=mu, maxval=sigma)
            params = jnp.eye(N=shape[0], M=shape[1]) * eps
        else: ## constant diagonal
            offDiagScale, _ = args
            params = jnp.eye(N=shape[0], M=shape[1]) * offDiagScale
    elif initType == "uniform": ## uniformly distributed values
        lb, ub = args
        params = random.uniform(dkey, shape, minval=lb, maxval=ub)
    elif initType == "fan_in_gaussian": ## fan-in scaled normal distributed values
        Phi = random.normal(dkey, shape)
        Phi = Phi * jnp.sqrt(1.0 / (shape[0] * 1.))
        params = Phi.astype(jnp.float32)
    elif initType == "gaussian" or initType == "normal": ## gaussian distributed values
        mu, sigma = args
        params = random.normal(dkey, shape) * sigma + mu
    elif initType == "constant": ## constant value(s)
        scale, _ = args
        params = jnp.ones(shape) * scale
    else:
        raise RuntimeError(
            "Initialization scheme (" + initType + ") is not recognized/supported!"
            )
    return params

