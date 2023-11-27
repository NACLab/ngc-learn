from ngclearn.engine.nodes.cells.cell import Cell
from jax import random, numpy as jnp, jit
#from ngclearn.engine.utils.math_utils import run_filter

## Poisson spike cell
class PoissCell(Cell):  # inherits from Node class
    """
    An approximate Poisson spike cell (using Bernoulli trial spikes), with
    optional temporal lag.

    Args:
        name: the string name of this cell

        n_units: number of cellular entities (neural population size)

        dt: integration time constant

        max_lag: lag coefficent; maximum time allowed to pass before first spike emitted
            (DEFAULT: 0)

        key: PRNG key to control determinism of any underlying synapses
            associated with this cell
    """
    def __init__(self, name, n_units, dt, max_lag=0., key=None, debugging=False):
        super().__init__(name, n_units, dt, key, debugging=debugging)
        self.max_lag = max_lag ## max time allowed to pass before first spike emitted

        if max_lag > 0.: ## compute lags if max_lag is > 0
            self.key, *subkeys = random.split(self.key, 2)
            self.lag = random.uniform(subkeys[0], (1, self.n_units), minval=0.,
                                      maxval=max_lag, dtype=jnp.float32)
        # cell compartments
        self.comp["in"] = None
        self.comp["s"] = None
        self.comp["tols"] = None

    def step(self):
        self.t = self.t + self.dt
        self.gather()
        lag_mask = 1.
        if self.max_lag > 0.: ## calc lag mask
            lag_mask = (self.lag < self.t).astype(jnp.float32)

        p_spk = self.comp["in"]  ## get probability of spike
        self.key, *subkeys = random.split(self.key, 2)
        s = random.bernoulli(subkeys[0], p=p_spk).astype(jnp.float32)
        self.comp["s"] = s * lag_mask
        if self.comp["tols"] is None:
            self.comp["tols"] = 0
        self.comp["tols"] = (1 - s) * self.comp["tols"] + (s * self.t)

    def custom_dump(self, node_directory, template=False):
        required_keys = ['max_lag']
        return {**super().custom_dump(node_directory, template),
                **{k: self.__dict__.get(k, None) for k in required_keys}}

    @staticmethod
    def get_default_out():
        """
        Returns the value within output compartment ``s``
        """
        return 's'

    comp_s = 's'
    comp_tols = 'tols'

class_name = PoissCell.__name__
