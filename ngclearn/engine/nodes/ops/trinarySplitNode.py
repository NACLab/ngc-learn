from ngclearn.engine.nodes.ops.op import Op
from jax import numpy as jnp, jit, vmap

@jit
@vmap
@vmap
def split(x):
    return jnp.array([jnp.abs(x) + x, (jnp.abs(x) - x)]) / 2

@jit
def extend_split(x):
    return jnp.reshape(split(x).T, [1, x.shape[1] * 2])

class TrinarySplitNode(Op):
    """
    A three-way splitting node; note that the output of this merges the
    results of the split into a compound vector (3x original dimensionality)

    Args:
        name: the string name of this operator

        n_units: number of calculating entities or units

        dt: integration time constant

        key: PRNG Key to control determinism of any underlying synapses
            associated with this operator
    """
    def __init__(self, name, n_units, dt, key=None, debugging=False):
        super().__init__(name, n_units, dt, key, debugging=debugging)

        # cell compartments
        self.comp["in"] = None
        self.comp["tols"] = None
        self.comp["s"] = None

    def step(self):
        self.t = self.t + self.dt
        self.gather()
        tri = self.comp["in"]
        spikes = extend_split(tri)
        self.comp['out'] = spikes
        self.comp['s'] = spikes
        self.comp["tols"] = (1 - spikes) * self.comp["tols"] + (spikes * self.t)

    def set_to_rest(self, batch_size=1, hard=True):
        if hard:
            super().set_to_rest(batch_size)
        else:
            self.comp['tols'] = jnp.zeros([batch_size, self.n_units])

    comp_tols = "tols"

class_name = TrinarySplitNode.__name__
