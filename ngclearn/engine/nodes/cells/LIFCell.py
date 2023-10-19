from jax import jit, numpy as jnp, random, nn
from functools import partial
from ngclearn.engine.nodes.cells.cell import Cell
import os, sys
################################################################################
from ngclearn.engine.utils.cell_utils import apply_lat_pressure, calc_snm_bool, \
                                             calc_snm_wta, calc_global_thr, \
                                             calc_percell_thr, calc_persist_percell_thr

## jit-i-fied LIF functionals

@jit
def integrate_volt(t, dt, j, v, tau_m, R_m):
    ## Run voltage ODE dynamics one step forward
    _v = v + (-v + j * R_m) * (dt/tau_m)
    ##_v = jnp.fmax(v_min, v + (-v + j) * (dt / tau_m))
    return _v

@partial(jit, static_argnums=[6])
def integrate_spk(t, dt, v, v_thr, v_reset, v_min, spk_fx):
    ## Run spike neuron model - get action potential
    s = spk_fx(v, v_thr)
    _v = v * (1. - s) + s * v_reset ## depolarize cells
    #_v = _v - s * v_thr ## depolarize cells
    _v = jnp.fmax(v_min, _v) ## impose lower bound on membrane potentials
    return _v, s

@partial(jit, static_argnums=[6])
def integrate_thr(t, dt, s, v_thr, thr_gain, thr_decay, thr_fx):
    ## Run threshold ODE dynamics one step forward
    _v_thr = thr_fx(s, v_thr, thr_gain, thr_decay)
    return _v_thr

@partial(jit, static_argnums=[7,10])
def run_cell(t, dt, j, v, v_thr, tau_m, R_m, spk_fx, v_reset, v_min, thr_fx,
             thr_gain, thr_decay):
    """
    Runs leaky integrator neuronal dynamics
    """
    ## Run cell's system of ODEs one step forward
    _v = integrate_volt(t, dt, j, v, tau_m, R_m)
    _v, s = integrate_spk(t, dt, _v, v_thr, v_reset, v_min, spk_fx)
    _v_thr = integrate_thr(t, dt, s, v_thr, thr_gain, thr_decay, thr_fx)
    return (_v, s, _v_thr)

## leaky integrate-and-fire cell
class LIFCell(Cell):  # inherits from Node class
    """
    A spiking cell based on the leaky integrate-and-fire (LIF) model. Also contains
    an adaptive threshold (per unit) scheme.

    Args:
        name: the string name of this cell

        n_units: number of cellular entities (neural population size)

        dt: integration time constant

        tau_m: membrane time constant

        v_thr_base: base value for adaptive thresholds (initial condition for
            per-cell thresholds)

        R_m: membrane resistance value

        thr_jitter: noise to apply to initial threshold values (one per unit)

        lat_Rinh: lateral modulation factor (DEFAULT: 0); if >0, this will trigger
            a heuristic form of lateral inhibition via a hollow matrix multiplication

        thr_mode: threshold function (Default: "dthr")

            :Note: values this can be are;
                "gthr" = persistent adaptive thresholds (a value per unit)
                "dthr" = local adaptive threshold (a value per unit)
                "gthr" = global adaptive threshold (one value for all units)

        thr_gain: how much adaptive thresholds increment by

        thr_decay: how much adaptive thresholds decrement/decay by

        spk_mode: spike response function (DEFAULT: "bool")

            :Note: values this can be are;
                "bool" = binary spikes;
                "wta" = winner-take-all spikes

        sign: scalar sign to multiply output signal by (DEFAULT: None)

        key: PRGN Key to control determinism of any underlying synapses
            associated with this cell
    """
    def __init__(self, name, n_units, dt, tau_m, v_thr_base, R_m=1., thr_jitter=0.1,
                 lat_Rinh=0., thr_mode='dthr', thr_gain=None, thr_decay=None,
                 spk_mode='bool', sign=None, key=None):
        super().__init__(name, n_units, dt, key)
        self.v_reset = -2. ## voltage value to set after action potential/spike
        self.v_min = -15. ## minimum voltage potential

        self.tau_m = tau_m  # membrane potential time constant
        self.R_m = R_m ## membrane resistance (mega-Ohms)
        self.sign = 1. if sign is None else sign
        self.lat_Rinh = lat_Rinh # if > 0, add hollow matrix heuristic for lateral inhibition
        self.v_thr_base = v_thr_base
        self.thr_mode = thr_mode # dthr = dynamic thresh, pthr = persist thresh
        self.thr_gain = 0. if thr_gain is None else thr_gain # 0.005
        self.thr_decay = 0. if thr_decay is None else thr_decay
        self.spk_mode = spk_mode # bool = inequality spike, wta = winner-take-all spike

        self.thr_fx = None
        match self.thr_mode:
            case 'pthr':
                self.thr_fx = calc_persist_percell_thr
            case 'dthr':
                self.thr_fx = calc_percell_thr
            case 'gthr':
                self.thr_fx = calc_global_thr
        self.spk_fx = None
        match self.spk_mode:
            case 'bool':
                self.spk_fx = calc_snm_bool
            case 'wta':
                self.spk_fx = calc_snm_wta

        # per-unit threshold values
        self.key, *subkeys = random.split(self.key, 2)
        jitter = thr_jitter #0.1 #0.035
        self.thr0 = v_thr_base + random.uniform(subkeys[0], (1, self.n_units),
                                                minval=-jitter, maxval=jitter,
                                                dtype=jnp.float32)

        ## heuristic lateral inhibition (via a Hollow matrix approximation)
        self.V = None ## hollow matrix
        self.lat_Rinh = lat_Rinh ## inhibitory resistance
        if self.lat_Rinh > 0.:
            self.key = random.PRNGKey(seed)
            self.key, *subkeys = random.split(self.key, 2)
            shape = (self.n_units, self.n_units)
            lb = 0.025 #0.01
            ub = 1. # 0.7
            MV = 1. - jnp.eye(self.n_units)
            self.V = random.uniform(subkeys[0], shape, minval=lb, maxval=ub, dtype=jnp.float32) * MV
            #self.V = jnp.ones(shape) * MV

        # cell compartments
        self.comp["j_inh"] = None
        self.comp["j"] = None  ## electrical current
        self.comp["v"] = None  ## voltage potential of nodes
        self.comp["s_prev"] = None ## record of previous spike at sim step t-1
        self.comp["s"] = None  ## current spike vector record
        self.comp["thr"] = self.thr0 + 0  ## per-unit threshold values
        self.comp["f(s)"] = None  ## signed spike
        self.comp["tols"] = None ## time of last spike

    def step(self):
        self.t = self.t + self.dt
        ## run cell dynamics one step forward
        self.gather()
        j = self.comp.get("j")
        v = self.comp.get("v")
        thr = self.comp.get("thr")

        if self.lat_Rinh > 0.:
            s = self.comp.get("s")
            j, j_inh = apply_lat_pressure(j, s, self.V, self.lat_Rinh)
            self.comp["j_inh"] = j_inh
        ## run LIF dynamics
        _v, _s, _thr = run_cell(self.t, self.dt, j, v, thr, self.tau_m, self.R_m,
                                self.spk_fx, self.v_reset, self.v_min,
                                self.thr_fx, self.thr_gain, self.thr_decay)
        self.comp["v"] = _v
        self.comp["s_prev"] = self.comp["s"] + 0 # store last spike record here
        self.comp["s"] = _s
        self.comp["out"] = _s
        self.comp["thr"] = _thr
        self.comp["f(s)"] = _s * self.sign
        self.comp["tols"] = (1 - _s) * self.comp["tols"] + (_s * self.t)

    def set_to_rest(self, batch_size=1, hard=True):
        if hard:
            thr_tmp = self.comp["thr"] + 0
            super().set_to_rest(batch_size)
            if self.thr_mode == "pthr":
                self.comp["thr"] = thr_tmp
            else:
                self.comp["thr"] = self.thr0 + 0

        else:
            self.comp['tols'] = jnp.zeros([batch_size, self.n_units])
        # if self.persist_thr == True:
        #     self.comp["thr"] = thr_tmp
        # else:
        #     self.comp["thr"] = self.thr0 + 0

    def custom_dump(self, node_directory, template=False) -> dict[str, any]:
        if not template:
            jnp.save(node_directory + "/thr0.npy", self.thr0)
            jnp.save(node_directory + "/thr.npy", self.comp["thr"])
        required_keys = ['tau_m', 'R_m', 'v_thr_base', 'thr_mode', 'thr_gain',
                         'thr_decay', 'spk_mode', 'sign', 'lat_Rinh']
        return {**super().custom_dump(node_directory, template),
                **{k: self.__dict__.get(k, None) for k in required_keys}}

    def custom_load(self, node_directory):
        if os.path.isfile(node_directory + "/thr0.npy"):
            self.thr0 = jnp.load(node_directory + "/thr0.npy")
        if os.path.isfile(node_directory + "/thr.npy"):
            self.comp["thr"] = jnp.load(node_directory + "/thr.npy")

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

class_name = LIFCell.__name__
