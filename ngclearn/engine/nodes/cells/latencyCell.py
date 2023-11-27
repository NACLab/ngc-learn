from ngclearn.engine.nodes.cells.cell import Cell
from jax import random, numpy as jnp, jit

## Latency encoding cell
class LatencyCell(Cell):  # inherits from Node class
    """
    A (nonlinear) latency encoding (spike) cell.

    Args:
        name: the string name of this cable

        n_units: number of cellular entities (neural population size)

        dt: integration time constant

        tau:

        thr:

        key: PRNG Key to control determinism of any underlying synapses
            associated with this cell
    """
    def __init__(self, name, n_units, dt, tau=5., thr=0.01, key=None, debugging=False):
        super().__init__(name, n_units, dt, key, debugging=debugging)
        self.tau = tau
        self.thr = thr
        self.linearize = False
        self.normalize = False

        # cell compartments
        self.comp["in"] = None
        self.comp["s"] = None
        self.comp["tols"] = None

    def step(self):
        self.t = self.t + self.dt
        self.gather()
        # tau_ = self.tau
        # if self.normalize == True:
        #     tau = num_steps - 1. - first_spike_time
        ## compute spike vector based on data-driven spike times
        data_in = self.comp["in"]
        if self.linearize == True: ## linearize spike time calculation
            stimes = jnp.fmax(-self.tau * (data_in - 1.), -self.tau * (self.thr - 1.))
            #torch.clamp_max((-tau * (data - 1)), -tau * (threshold - 1))
        else: ## standard nonlinear spike time calculation
            stimes = jnp.log(data_in / (data_in - self.thr)) * self.tau ## calc spike times
        s = (stimes <= self.t).astype(jnp.float32) # get spike
        self.comp["s"] = s
        self.comp["tols"] = (1 - s) * self.comp["tols"] + (s * self.t)

    def custom_dump(self, node_directory, template=False):
        required_keys = ['tau', 'thr']
        return {**super().custom_dump(node_directory, template),
                **{k: self.__dict__.get(k, None) for k in required_keys}}

    @staticmethod
    def get_default_out():
        """
        Returns the value within output compartment ``s``
        """
        return 's'

class_name = LatencyCell.__name__
