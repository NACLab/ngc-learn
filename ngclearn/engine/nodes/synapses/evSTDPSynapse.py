## synapse that learns via post-synaptic event-based STDP
from ngclearn.engine.nodes.synapses.synapse import Synapse
from jax import numpy as jnp, jit, random
from functools import partial
import os

#@jit
@partial(jit, static_argnums=6)
def _evolve(pre, post, W, w_bound=1., eta=0.00005, lamb=0., w_norm=None):
    pos_shift = w_bound - W * (1. + lamb) # this follows rule in eqn 18 of paper
    neg_shift = -W * (1. + lamb)
    dW = jnp.where(pre.T, pos_shift, neg_shift)
    dW = (dW * post) * eta #* (1. - w) * eta
    _W = W + dW
    if w_norm is not None:
       _W = _W * (w_norm/(jnp.linalg.norm(_W, axis=1, keepdims=True) + 1e-5)) #jnp.clip_by_norm(_w, 1.)
    _W = jnp.clip(_W, 0.01, w_bound) # not in source paper
    return _W

@jit
def run_synapse(inp, W, sign):
    out = jnp.matmul(inp, W)
    return out * sign

class EvSTDPSynapse(Synapse):  # inherits from Node class
    """
    A synaptic transform that supports event-driven STDP plasticity
    from reference/inspiration:
    | Tavanaei, Amirhossein, Timothée Masquelier, and Anthony Maida.
    | "Representation learning using event-based STDP." Neural Networks 105
    | (2018): 294-303.

    Args:
        name: the string name of this cable (Default = None which creates an auto-name)

        dt: integration time constant

        shape: tensor shape of this synapse

        lamb: ev-STDP lambda coefficient

        eta: "learning rate" or step-size to modulate plasticity adjustment by

        w_norm: Frobenius norm constraint value to apply after synaptic matrix
            update

        sign: scalar sign to multiply output signal by (DEFAULT: 1)

        key: PRNG Key to control determinism of any underlying synapses
            associated with this cable
    """
    def __init__(self, name, dt, shape, lamb, eta, w_norm=None,
                 sign=None, key=None):
        super().__init__(name=name, shape=shape, dt=dt, key=key)
        self.eta = eta
        self.lamb = lamb
        self.sign = 1 if sign is None else sign
        self.w_bound = 1. ## soft weight constraint
        self.w_norm = w_norm ## normalization constant for synaptic matrix after update

        # cell compartments
        self.comp["in"] = None
        self.comp["pre"] = None
        self.comp["post"] = None

        # preprocessing - set up synaptic efficacy matrix
        self.key, *subkeys = random.split(self.key, 2)
        lb = 0.025 # 0.25
        ub = 0.8 #1. #0.5 # 0.75
        self.W = random.uniform(subkeys[0], self.shape, minval=lb, maxval=ub, dtype=jnp.float32)

    def step(self):
        self.gather()
        i = self.comp.get("in") ## get input to synaptic projection
        self.comp['out'] = run_synapse(i, self.W, self.sign)

        self.t = self.t + self.dt

    def evolve(self):
        self.gather()
        _pre = self.comp['pre']
        _post = self.comp['post']

        self.W = _evolve(_pre, _post, self.W, self.w_bound, self.eta, self.lamb,
                         self.w_norm)

    def custom_dump(self, node_directory, template=False) -> dict[str, any]:
        if not template:
            jnp.save(node_directory + "/W.npy", self.W)
        required_keys = ['shape', 'lamb', 'sign', 'eta', 'w_norm']
        return {k: self.__dict__.get(k, None) for k in required_keys}

    def custom_load(self, node_directory):
        if os.path.isfile(node_directory + "/W.npy"):
            self.W = jnp.load(node_directory + "/W.npy")

class_name = EvSTDPSynapse.__name__
