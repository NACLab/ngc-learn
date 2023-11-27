from jax import jit, numpy as jnp, random, nn
from functools import partial
from ngclearn.engine.nodes.cells.cell import Cell
import os
################################################################################
from ngclearn.engine.utils.math_utils import softmax
from ngclearn.engine.utils.cell_utils import calc_global_thr

## jit-i-fied WTA-spike-response functionals

#@jit #
@partial(jit, static_argnums=7)
def run_wta_score(t, j, v, v_thr, dt, R_m, thr_gain, hard_max_spk=False):
    """
    winner-take-all (WTA) score neuronal dynamics
    """
    _v = j * R_m
    vp = softmax(_v) # convert to Categorical (spike) probabilities
    if hard_max_spk == True: ## forces a strict WTA single spike
        s = nn.one_hot(jnp.argmax(vp, axis=1), j.shape[1])
    else:
        s = (vp > v_thr).astype(jnp.float32) ## calculate action potential
    _v_thr = calc_global_thr(s, v_thr, thr_gain, q=1., vthr_min=0.05)
    return (vp, s, _v_thr)

## WTAS cell
class WTASCell(Cell):  # inherits from Node class
    """
    A winner-take-all score (WTAS) cell. This neuron guarantees that only
    one neuron will spike at any time step (and is considered to be a very
    simple spike-response model).

    Reference:
    Tavanaei, Amirhossein, Timothée Masquelier, and Anthony Maida.
    "Representation learning using event-based STDP." Neural Networks 105
    (2018): 294-303.

    Args:
        name: the string name of this cell

        n_units: number of cellular entities (neural population size)

        dt: integration time constant

        v_thr_base: base value for adaptive thresholds (initial condition for
            per-cell thresholds)

        R_m: membrane resistance value

        thr_gain: how much adaptive thresholds increment by

        thr_jitter: noise to apply to initial threshold values (one per unit)

        sign: scalar sign to multiply output signal by (DEFAULT: None)

        key: PRNG key to control determinism of any underlying synapses
            associated with this cell
    """
    def __init__(self, name, n_units, dt, v_thr_base, R_m=1., thr_gain=None,
                 thr_jitter=0., sign=None, key=None, debugging=False):
        super().__init__(name, n_units, dt, key, debugging=debugging)
        self.sign = 1. if sign is None else sign
        self.v_thr_base = v_thr_base
        self.thr_gain = 0. if thr_gain is None else thr_gain
        self.R_m = R_m ## membrane resistance (mega-Ohms)

        # per-unit threshold values
        self.key, *subkeys = random.split(self.key, 2)
        jitter = thr_jitter #0.05
        #self.thr0 = v_thr_base
        self.thr0 = v_thr_base + random.uniform(subkeys[0], (1, self.n_units),
                                                minval=-jitter, maxval=jitter,
                                                dtype=jnp.float32)

        # cell compartments
        self.comp["j"] = None  ## electrical current
        self.comp["v"] = None  ## voltage potential of nodes
        self.comp["s_prev"] = None ## record of previous spike at sim step t-1
        self.comp["s"] = None  ## current spike vector record
        self.comp["thr"] = self.thr0 + 0  ## per-unit threshold values
        #self.comp["z"] = None  ## trace variable
        self.comp["f(s)"] = None  ## signed spike
        #self.comp["f(z)"] = None  ## signed trace
        self.comp["tols"] = None ## time of last spike

    def step(self):
        ## run cell dynamics one step forward
        self.gather()
        j = self.comp.get("j")
        v = self.comp.get("v")
        thr = self.comp.get("thr")

        ## run LIF dynamics
        _v, _s, _thr = run_wta_score(self.t, j, v, thr, self.dt, self.R_m,
                                     self.thr_gain)

        self.comp["v"] = _v
        self.comp["s_prev"] = self.comp["s"] + 0 # store last spike record here
        self.comp["s"] = _s
        self.comp["out"] = _s
        self.comp["thr"] = _thr
        self.comp["f(s)"] = _s * self.sign
        self.comp["tols"] = (1 - _s) * self.comp["tols"] + (_s * self.t)
        self.t = self.t + self.dt

    def set_to_rest(self, batch_size=1, hard=True):
        if hard:
            super().set_to_rest(batch_size)
            self.comp["thr"] = self.thr0 + 0
        else:
            self.comp['tols'] = jnp.zeros([batch_size, self.n_units])
    def custom_dump(self, node_directory, template=False) -> dict[str, any]:
        if not template:
            jnp.save(node_directory + "/thr0.npy", self.thr0)
        required_keys = ['sign', 'v_thr_base', 'R_m', 'thr_gain', 'thr_jitter']
        return {**super().custom_dump(node_directory, template),
                **{k: self.__dict__.get(k, None) for k in required_keys}}

    def custom_load(self, node_directory):
        if os.path.isfile(node_directory + "/thr0.npy"):
            self.thr0 = jnp.load(node_directory + "/thr0.npy")

    @staticmethod
    def get_default_in():
        """
        Returns the value within input compartment ``j``
        """
        return 'j'

    @staticmethod
    def get_default_out():
        """
        Returns the value within output compartment ``s``
        """
        return 's'

    comp_j = "j"
    comp_v = "v"
    comp_s_prev = "s_prev"
    comp_s = "s"
    comp_thr = "thr"
    comp_fs = "fs"
    comp_tols = "tols"

class_name = WTASCell.__name__
