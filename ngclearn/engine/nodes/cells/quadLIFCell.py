from jax import jit, numpy as jnp, random, nn
from functools import partial
from ngclearn.engine.nodes.cells.LIFCell import LIFCell
import os, sys
################################################################################
from ngclearn.engine.utils.cell_utils import apply_lat_pressure, calc_snm_bool, \
                                             calc_snm_wta, calc_global_thr, \
                                             calc_percell_thr, calc_persist_percell_thr

## jit-i-fied quadratic-LIF functionals

@jit
def integrate_volt(t, dt, j, v, tau_m, R_m, v_c, a0):
    v_rest = 0. # we fix this to 0 for you
    ## Run voltage ODE dynamics one step forward
    _v = v + ((v - v_rest) * (v - v_c) * a0 + j * R_m) * (dt/tau_m)
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

@partial(jit, static_argnums=[9,12])
def run_cell(t, dt, j, v, v_thr, tau_m, R_m, a0, v_c, spk_fx, v_reset, v_min,
             thr_fx, thr_gain, thr_decay):
    """
    quadratic leaky integrator neuronal dynamics
    """
    ## Run cell's system of ODEs one step forward
    _v = integrate_volt(t, dt, j, v, tau_m, R_m, v_c, a0)
    _v, s = integrate_spk(t, dt, _v, v_thr, v_reset, v_min, spk_fx)
    _v_thr = integrate_thr(t, dt, s, v_thr, thr_gain, thr_decay, thr_fx)
    return (_v, s, _v_thr)

## quadratic leaky integrate-and-fire cell
class QuadLIFCell(LIFCell):  # inherits from LIFCell class
    """
    A spiking cell based on the quadratic leaky integrate-and-fire (QuadLIF) model.
    Also contains an adaptive threshold (per unit) scheme.

    Note that the quadratic LIF's cellular dynamics proceeds according to:
    | d.Vz/d.t = a0 * (V - V_rest) * (V - V_c) + Jz * R) * (dt/tau_mem)

    where:
    |   a0 - scaling factor for voltage accumulation
    |   V_c - critical voltage - 0.8 (chosen heuristically)

    Ref:  https://neuronaldynamics.epfl.ch/online/Ch5.S3.html

    Args:
        name: the string name of this cell

        n_units: number of cellular entities (neural population size)

        dt: integration time constant

        tau_m: membrane time constant

        v_thr_base: base value for adaptive thresholds (initial condition for
            per-cell thresholds)

        R_m: membrane resistance value

        thr_jitter: noise to apply to initial threshold values (one per unit)

        a0: voltage accumulation factor

        v_c: critical voltage value

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

        key: PRNG key to control determinism of any underlying synapses
            associated with this cell
    """
    def __init__(self, name, n_units, dt, tau_m, v_thr_base, R_m=1., thr_jitter=0.1,
                 a0=1., v_c=0.2, lat_Rinh=0., thr_mode='dthr', thr_gain=None,
                 thr_decay=None, spk_mode='bool', sign=None, key=None, debugging=False):
        super().__init__(name, n_units, dt, tau_m, v_thr_base, R_m, thr_jitter,
                         lat_Rinh, thr_mode, thr_gain, thr_decay, spk_mode, sign,
                         key, debugging=debugging)
        self.a0 = a0 #1. # scaling factor
        self.v_c = v_c # 0.2 or 0.8 # critical voltage
        #self.v_reset = -0.5
        #self.v_min = -1.5

    def step(self):
        self.t = self.t + self.dt
        self.gather()

        j = self.comp.get("j")
        v = self.comp.get("v")
        thr = self.comp.get("thr")

        if self.lat_Rinh > 0.:
            s = self.comp.get("s")
            j, j_inh = apply_lat_pressure(j, s, self.V, self.lat_Rinh)
            self.comp["j_inh"] = j_inh
        ## run Quadratic-LIF dynamics
        _v, _s, _thr = run_cell(self.t, self.dt, j, v, thr, self.tau_m, self.R_m,
                                self.a0, self.v_c, self.spk_fx, self.v_reset,
                                self.v_min, self.thr_fx, self.thr_gain,
                                self.thr_decay)
        self.comp["v"] = _v
        self.comp["s_prev"] = self.comp["s"] + 0 # store last spike record here
        self.comp["s"] = _s
        self.comp["out"] = _s
        self.comp["thr"] = _thr
        self.comp["f(s)"] = _s * self.sign
        self.comp["tols"] = (1 - _s) * self.comp["tols"] + (_s * self.t)

    def custom_dump(self, node_directory, template=False) -> dict[str, any]:
        if not template:
            jnp.save(node_directory + "/thr0.npy", self.thr0)
            jnp.save(node_directory + "/thr.npy", self.comp["thr"])
        required_keys = ['v_c', 'a0']
        return {**super().custom_dump(node_directory, template),
                **{k: self.__dict__.get(k, None) for k in required_keys}}

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
    
class_name = QuadLIFCell.__name__
