## pass-through cell
from ngclearn.engine.nodes.synapses.synapse import Synapse
from jax import numpy as jnp, jit, random
import os

@jit
def run_synapse(inp, W, sign):
    return jnp.matmul(inp, W) * sign

class ProjectionSynapse(Synapse):  # inherits from Cell class
    """
    A synaptic transform that transforms signals that travel across
    via a clustered set of synapses. Note that this synaptic node does NOT
    support any form of plasticity.

    Args:
        name: the string name of this cable

        dt: integration time constant

        shape: tensor shape of this synapse

        sign: scalar sign to multiply output signal by (DEFAULT: 1)

        seed: integer seed to control determinism of any underlying synapses
            associated with this cable
    """
    def __init__(self, name, dt, shape, sign=None, seed=69):
        super().__init__(name=name, shape=shape, dt=dt, seed=seed)
        self.shape = shape  # shape of synaptic matrix W
        self.sign = 1 if sign is None else sign

        # cell compartments
        self.comp["in"] = None

        #Preprocessing
        self.key = random.PRNGKey(seed)
        self.key, *subkeys = random.split(self.key, 2)
        sigma = 0.025
        self.W = random.normal(subkeys[0], self.shape, dtype=jnp.float32) * sigma

    def step(self):
        self.gather()
        # x = self.comp.get("in") # get input stimulus
        i = self.comp.get("in")
        self.comp['out'] = run_synapse(i, self.W, self.sign)
        self.t = self.t + self.dt

    def custom_dump(self, node_directory, template=False) -> dict[str, any]:
        if not template:
            jnp.save(node_directory + "/W.npy", self.W)
        required_keys = ['shape', 'sign']
        return {k: self.__dict__.get(k, None) for k in required_keys}

    def custom_load(self, node_directory):
        if os.path.isfile(node_directory + "/W.npy"):
            self.W = jnp.load(node_directory + "/W.npy")

class_name = ProjectionSynapse.__name__
