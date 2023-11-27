## pass-through cell
from ngclearn.engine.nodes.synapses.synapse import Synapse
from jax import numpy as jnp, jit, random
import os

@jit
def calc_adjustment(pre, pre_gate, post, post_gate, W, w_bound): ## calc scaled synaptic matrix delta
    dW = jnp.matmul((pre * pre_gate).T, (post * post_gate)) * (w_bound - W)
    return dW

@jit
def evolve(pre, pre_gate, post, post_gate, W, w_bound, eta): ## run step of gradient ascent
    dW = calc_adjustment(pre, pre_gate, post, post_gate, W, w_bound)
    _W = W + dW * eta
    _W = jnp.clip(_W, 0., w_bound)
    return _W

@jit
def evolve_2sided(pre, pre_gate, post, post_gate, W, w_bound, eta): ## run step of gradient ascent (for neg/pos synapses)
    dW = jnp.matmul((pre * pre_gate).T, (post * post_gate)) * (w_bound - jnp.abs(W))
    _W = W + dW * eta
    _W = jnp.clip(_W, -w_bound, w_bound)
    return _W


@jit
def run_synapse(inp, W, sign):
    out = jnp.matmul(inp, W)
    return out * sign

class HebbianSynapse(Synapse):  # inherits from Cell class
    """
    A synaptic transform that transforms signals that travel across via a
    bundle of synapses and adapts based on local multi-factor Hebbian plasticity.

    Args:
        name: the string name of this cable (Default = None which creates an auto-name)

        dt: integration time constant

        shape: tensor shape of this synapse

        eta: "learning rate" or step-size to modulate plasticity adjustment by

        sign: scalar sign to multiply output signal by (DEFAULT: 1)

        seed: integer seed to control determinism of any underlying synapses
            associated with this cable
    """
    def __init__(self, name, dt, shape, eta, sign=None, key=None, debugging=False):
        super().__init__(name=name, shape=shape, dt=dt, key=key, debugging=debugging)
        self.shape = shape  # shape of synaptic matrix W
        self.sign = 1 if sign is None else sign
        self.eta = eta ## step size
        self.w_bound = 1. ## soft weight constraint
        self.use_2sided_rule = True #False
        # cell compartments
        self.comp["in"] = None

        #Preprocessing
        self.key, *subkeys = random.split(self.key, 2)
        sigma = 0.025
        self.W = random.normal(subkeys[0], self.shape, dtype=jnp.float32) * sigma

        self.evo = evolve_2sided if self.use_2sided_rule else evolve


    def step(self):
        self.gather()
        # x = self.comp.get("in") # get input stimulus
        i = self.comp.get("in")
        self.comp['out'] = run_synapse(i, self.W, self.sign)

        self.t = self.t + self.dt

    def evolve(self):
        self.gather()
        pre_gate = self.comp.get("pre_gate", 1)
        post_gate = self.comp.get("post_gate", 1)

        pre = self.comp.get('pre')
        post = self.comp.get('post')


        self.W = self.evo(pre, pre_gate, post, post_gate, self.W, self.w_bound, self.eta)


    def custom_dump(self, node_directory, template=False) -> dict[str, any]:
        if not template:
            jnp.save(node_directory + "/W.npy", self.W)
        required_keys = ['shape', 'eta', 'sign']
        return {k: self.__dict__.get(k, None) for k in required_keys}

    def custom_load(self, node_directory):
        if os.path.isfile(node_directory + "/W.npy"):
            self.W = jnp.load(node_directory + "/W.npy")


class_name = HebbianSynapse.__name__
