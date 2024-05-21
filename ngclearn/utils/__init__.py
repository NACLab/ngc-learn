import jax.numpy as jnp

def tensorstats(tensor):
    if tensor is not None:
        _tensor = jnp.asarray(tensor)
        return {
            'mean': _tensor.mean(),
            'std': _tensor.std(),
            'mag': jnp.abs(_tensor).max(),
            'min': _tensor.min(),
            'max': _tensor.max(),
        }
    else:
        return {
            'mean': None,
            'std': None,
            'mag': None,
            'min': None,
            'max': None,
        }