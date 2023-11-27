## synapse that learns via reward-modulated (trace-based) STDP
from ngclearn.engine.nodes.synapses.synapse import Synapse
from ngclearn.engine.utils.synapse_utils import pressure
from jax import numpy as jnp, jit, random
from functools import partial
import os

@partial(jit, static_argnums=[9,10])
def calc_update(dt, pre, x_pre, post, x_post, W, w_bound=1., x_tar=0.7,
                Aplus=1., Aminus=0., exp_beta=None, mu=1.):
    ## calculate post-synaptic STDP Hebbian adjustment
    if exp_beta is not None: ## exp-depend STDP
        post_term1 = jnp.exp(-exp_beta * W) * jnp.matmul(x_pre.T, post)
        x_tar_vec = x_pre * 0 + x_tar # need to broadcast scalar x_tar to mat/vec form
        post_term2 = jnp.exp(-exp_beta * (w_bound - W)) * jnp.matmul(x_tar_vec.T, post)
        dWpost = (post_term1 - post_term2) * Aplus
        dWpre = 0.
        if Aminus > 0.:
            dWpre = -jnp.exp(-exp_beta * W) * jnp.matmul(pre.T, x_post) * Aminus
    else: ## power-law STDP, mu = 1 (fixed)
        dWpost = jnp.matmul((x_pre - x_tar).T, post) * (w_bound - W) * Aplus
        dWpre = 0.
        if Aminus > 0.:
            dWpre = -(jnp.matmul(pre.T, x_post)) * W * Aminus
    dW = dWpost + dWpre
    return dW

@jit
def _update_egTrace(dt, dW, Eg, tau_e):
    ## update eligibility trace matrix
    _Eg = Eg + (-Eg + dW) * (dt / tau_e)
    return _Eg

#@partial(jit, static_argnums=[6,9])
def _evolve_synapses(dt, W, Eg, r, dH, EgMask, tau_w=0., w_bound=1.,
                     eta=0.00005, w_norm=None):
    ## physically adjust the synaptic efficacies in W
    # dH (if a matrix) triggers 3-factor rule:  Eligib * Hebb * Modulator
    if tau_w > 0.:
        dWraw = Eg * dH * EgMask
        dWraw = jnp.where(dWraw >= 0., dWraw * r, dWraw)
        dW = -W * (dt / tau_w) + (dWraw * eta) # apply ODE
        #dW = ( -W + (Eg * dH * r) ) * (dt / tau_w) # apply ODE
        #dW = dW * eta
    else:
        dWraw = Eg * dH * EgMask
        dWraw = jnp.where(dWraw >= 0., dWraw * r, dWraw)
        # pos = (dWraw >= 0.).astype(jnp.float32)
        # dWraw = (dWraw * pos) * r + (dWraw * (1. - pos))
        dW = dWraw * eta # apply raw rule
    _W = W + dW
    if w_norm is not None:
        _W = _W * (w_norm/(jnp.linalg.norm(_W, axis=1, keepdims=True) + 1e-5))
    _W = jnp.clip(_W, 0.01, w_bound) # not in source paper
    return _W

@jit
def run_synapse(inp, W, sign):
    out = jnp.matmul(inp, W)
    return out * sign

class RSTDPSynapse(Synapse):  # inherits from Node class
    """
    Reward-modulated spike-timing-dependent plasticity (R-STDP) with a
    configurable modulatory factor.

    | References/inspiration:
    | Florian, Răzvan V. "Reinforcement learning through modulation of
    | spike-timing-dependent synaptic plasticity." Neural computation 19.6
    | (2007): 1468-1502.

    Args:
        name: the string name of this cable (Default = None which creates an auto-name)

        dt: integration time constant

        shape: tensor shape of this synapse

        eta: "learning rate" or step-size to modulate plasticity adjustment by

        tau_e: eligibility trace time constant

        tau_w: synaptic decay time constant

        x_tar: controls degree of pre-synaptic disconnect, i.e., amount of
            decay (higher leads to lower synaptic values) (DEFAULT: 0.7)

        use_td_error:  use temporal-difference based modulation

        reset_Elg_on_reward: resets the eligibility matrix to 0 upon application
            of a reward signal

        Aplus: strength of long-term potentiation (LTP)

        Aminus: strength of long-term depression (LTD)

        exp_beta: controls effect of Hebbian shift for exponential STPD
            (Default: None)

        w_norm: Frobenius norm constraint value to apply after synaptic matrix update

        push_rate:

        push_amount:

        sign: scalar sign to multiply output signal by (DEFAULT: 1)

        key: PRNG Key to control determinism of any underlying synapses
            associated with this cable
    """
    def __init__(self, name, dt, shape, eta, tau_e=100., tau_w=1000.,
                 x_tar=0.0, use_td_error=True, reset_Elg_on_reward=True,
                 Aplus=1., Aminus=0., exp_beta=None, w_norm=None, push_rate=-1,
                 push_amount=0., sign=None, key=None, debugging=False):
        super().__init__(name=name, shape=shape, dt=dt, key=key, debugging=debugging)
        self.eta = eta
        # STDP meta-parameters
        self.x_tar = x_tar
        self.Aplus = Aplus
        self.Aminus = Aminus
        self.exp_beta = exp_beta
        # reward modulation meta-parameters
        self.tau_e = tau_e ## time constant for eligibility trace ODE
        self.tau_w = tau_w ## time constant for synaptic change ODE
        self.shape = shape  # shape of synaptic matrix W
        self.sign = 1 if sign is None else sign
        self.w_bound = 1. ## soft weight constraint
        self.w_norm = w_norm ## normalization constant for synaptic matrix after update
        self.reset_Elg_on_reward = reset_Elg_on_reward
        self.use_td_error = use_td_error # TD works better than raw reward...
        # orthogonalization meta-parameters
        self.push_rate = push_rate
        self.push_amount = push_amount

        # cell compartments
        self.comp["in"] = None
        self.comp["pre"] = None
        self.comp["x_pre"] = None
        self.comp["post"] = None
        self.comp["x_post"] = None
        self.comp["r"] = None ## reward compartment
        self.comp["Eg_trace"] = None ## eligibility trace matrix
        self.comp["rSum"] = None
        self.comp['rN'] = None
        self.comp['EgMask'] = None

        # preprocessing - set up synaptic efficacy matrix
        self.key, *subkeys = random.split(self.key, 2)
        lb = 0.025 # 0.25
        ub = 0.55 #0.8
        self.W = random.uniform(subkeys[0], self.shape, minval=lb, maxval=ub, dtype=jnp.float32)

    def step(self):
        self.gather()
        i = self.comp.get("in") ## get input to synaptic projection
        self.comp['out'] = run_synapse(i, self.W, self.sign)

        self.t = self.t + self.dt

    def evolve(self):
        self.gather()
        Eg = self.comp['Eg_trace'] ## get eligibility trace
        r = self.comp['r'] ## get reward (trigger)
        EgMask = self.comp['EgMask']
        if EgMask is None:
            EgMask = 1.

        _pre = self.comp['pre'] ## pre-synaptic spike
        _x_pre = self.comp['x_pre'] ## pre-synaptic trace
        _post = self.comp['post'] ## post-synaptic spike
        _x_post = self.comp['x_post'] ## post-synaptic trace
        dW = calc_update(self.dt, _pre, _x_pre, _post, _x_post, self.W,
                         self.w_bound, self.x_tar, self.Aplus, self.Aminus,
                         self.exp_beta)
        ## decide what to do w/ eligibility trace
        if r is None:
            ## update eligibility trace
            Eg = _update_egTrace(self.dt, dW, Eg, self.tau_e)
            self.comp['Eg_trace'] = Eg
        else:
            ## Compute modulatory signal
            if self.use_td_error == True: ## modulate via TD error
                r_sum = self.comp['rSum']
                rN = self.comp['rN']
                if rN == 0.:
                    rN = 1.
                M = r - (r_sum/rN) # reward - expected reward
                self.comp['rSum'] = r_sum + r
                self.comp['rN'] += 1.
            else: ## modulate via raw reward
                M = r
            ## Directly adjust synaptic weights (assuming trace is up-to-date)
            Eg = _update_egTrace(self.dt, dW, Eg, self.tau_e)
            dH = 1. #dH = dW
            self.W = _evolve_synapses(self.dt, self.W, Eg, M, dH, EgMask,
                                      self.tau_w, self.w_bound, self.eta,
                                      self.w_norm)
            if self.push_rate > 0: ## apply orthogonalization process to W
                self.key, subkey = random.split(self.key, 2)
                self.W = pressure(self.W, key=subkey, a=self.push_amount,
                                  min_val=0.01, max_val=1)
            if self.reset_Elg_on_reward == True:
                self.comp['Eg_trace'] = Eg * 0
            self.comp['r'] = None
            self.comp['EgMask'] = None

    def set_to_rest(self, batch_size=1, hard=True):
        if hard:
            super().set_to_rest(batch_size)
            self.comp['Eg_trace'] = self.W * 0 # reset eligibility matrix
            self.comp['r'] = None
            self.comp['EgMask'] = None
            self.comp['rSum'] = 0.
            self.comp['rN'] = 0.

    def custom_dump(self, node_directory, template=False) -> dict[str, any]:
        if not template:
            jnp.save(node_directory + "/W.npy", self.W)
        required_keys = ['shape', 'tau_e', 'tau_w', 'x_tar', 'sign', 'eta',
                         'Aplus', 'Aminus', 'exp_beta', 'w_norm', 'use_td_error',
                         'reset_Elg_on_reward', 'push_rate', 'push_amount']
        return {k: self.__dict__.get(k, None) for k in required_keys}

    def custom_load(self, node_directory):
        if os.path.isfile(node_directory + "/W.npy"):
            self.W = jnp.load(node_directory + "/W.npy")

    comp_pre = "pre"
    comp_x_pre = "x_pre"
    comp_post = "post"
    comp_x_post = "x_post"
    comp_r = "r"
    comp_Eg_trace = "Eg_trace"
    comp_rSum = "rSum"
    comp_rN = "rN"
    comp_EgMask = "EgMask"

class_name = RSTDPSynapse.__name__
