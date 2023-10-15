## synapse that learns via trace-based STDP
from ngclearn.engine.nodes.synapses.synapse import Synapse
from jax import numpy as jnp, jit, random
from functools import partial
import os

@partial(jit, static_argnums=[9,10,11])
def _evolve_expSTDP(pre, x_pre, post, x_post, W, w_bound=1., eta=0.00005,
                    x_tar=0.7, exp_beta=1., Aplus=1., Aminus=0., w_norm=None):
    ## equations 4 from Diehl and Cook - full exponential weight-dependent STDP
    ## calculate post-synaptic term
    post_term1 = jnp.exp(-exp_beta * W) * jnp.matmul(x_pre.T, post)
    x_tar_vec = x_pre * 0 + x_tar # need to broadcast scalar x_tar to mat/vec form
    post_term2 = jnp.exp(-exp_beta * (w_bound - W)) * jnp.matmul(x_tar_vec.T, post)
    dWpost = (post_term1 - post_term2) * Aplus
    ## calculate pre-synaptic term
    dWpre = 0.
    if Aminus > 0.:
        dWpre = -jnp.exp(-exp_beta * W) * jnp.matmul(pre.T, x_post) * Aminus

    ## calc final weighted adjustment
    dW = (dWpost + dWpre) * eta
    _W = W + dW
    if w_norm is not None:
        _W = _W * (w_norm/(jnp.linalg.norm(_W, axis=1, keepdims=True) + 1e-5))
    _W = jnp.clip(_W, 0.01, w_bound) # not in source paper
    return _W

@partial(jit, static_argnums=[9,10,11])
def _evolve_powerLawSTDP(pre, x_pre, post, x_post, W, w_bound=1., eta=0.00005,
                         x_tar=0.7, mu=1.1, Aplus=1., Aminus=0., w_norm=None):
    ## equations 3, 5, & 6 from Diehl and Cook - full power-law STDP
    post_shift = jnp.power(w_bound - W, mu)
    pre_shift = jnp.power(W, mu)

    ## calculate post-synaptic term
    #dWpost = (shift * (x_pre - x_tar).T * post) * Aplus #(shift * post) * (x_pre - x_tar).T
    dWpost = (post_shift * jnp.matmul((x_pre - x_tar).T, post)) * Aplus
    ## calculate pre-synaptic term
    dWpre = 0.
    if Aminus > 0.:
        #dWpre = -(W * (x_post * pre.T)) * Aminus
        dWpre = -(pre_shift * jnp.matmul(pre.T, x_post)) * Aminus

    ## calc final weighted adjustment
    dW = (dWpost + dWpre) * eta
    _W = W + dW
    if w_norm is not None:
        _W = _W * (w_norm/(jnp.linalg.norm(_W, axis=1, keepdims=True) + 1e-5))
    _W = jnp.clip(_W, 0.01, w_bound) # not in source paper
    return _W

@jit
def run_synapse(inp, W, sign):
    out = jnp.matmul(inp, W)
    return out * sign

class TrSTDPSynapse(Synapse):  # inherits from Node class
    """
    A synaptic transform supports trace-based spike-timing-dependent
    plasticity (STDP) from (Diehl and Cook, 2015), i.e., trace form of the
    event-driven form of Bi & Poo's original STDP w/ synaptic decay. Two
    specific variants of trace-based STDP are possible:
    | 1) if exp_beta is not None (good value is 1.0), exponential STDP will be used,
    | 2) if exp_beta is None and mu > 0., power law STDP will be used

    | References/inspiration:
    | Bi, Guo-qiang, and Mu-ming Poo. "Synaptic modification by correlated
    | activity: Hebb's postulate revisited." Annual review of neuroscience 24.1
    | (2001): 139-166.

    | Diehl, Peter U., and Matthew Cook. "Unsupervised learning of digit
    | recognition using spike-timing-dependent plasticity." Frontiers in
    | computational neuroscience 9 (2015): 99.

    Args:
        name: the string name of this cable (Default = None which creates an auto-name)

        dt: integration time constant

        shape: tensor shape of this synapse

        eta: "learning rate" or step-size to modulate plasticity adjustment by

        mu: controls the power scale of the Hebbian shift for power law STDP
            (DEFAULT: 1)

        exp_beta: controls effect of Hebbian shift for exponential STPD
            (Default: None)

        x_tar: controls degree of pre-synaptic disconnect, i.e., amount of
            decay (higher leads to lower synaptic values) (DEFAULT: 0.7)

        Aplus: strength of long-term potentiation (LTP)

        Aminus: strength of long-term depression (LTD)

        w_norm: Frobenius norm constraint value to apply after synaptic matrix 
            update

        sign: scalar sign to multiply output signal by (DEFAULT: 1)

        seed: integer seed to control determinism of any underlying synapses
            associated with this cable
    """
    def __init__(self, name, dt, shape, eta, mu=1., exp_beta=None,
                 x_tar=0.7, Aplus=1., Aminus=0., w_norm=None, sign=None, seed=69):
        super().__init__(name=name, shape=shape, dt=dt, seed=seed)
        self.eta = eta
        self.mu = mu ## power to raise STDP adjustment by
        self.exp_beta = exp_beta ## if not None, will trigger exp-depend STPD rule
        self.x_tar = x_tar ## target (pre-synaptic) trace activity value
        self.Aplus = Aplus ## LTP strength
        self.Aminus = Aminus ## LTD strength
        self.shape = shape  # shape of synaptic matrix W
        self.sign = 1 if sign is None else sign
        self.w_bound = 1. ## soft weight constraint
        self.w_norm = w_norm ## normalization constant for synaptic matrix after update

        # cell compartments
        self.comp["in"] = None
        self.comp["pre"] = None
        self.comp["x_pre"] = None
        self.comp["post"] = None
        self.comp["x_post"] = None

        # preprocessing - set up synaptic efficacy matrix
        self.key = random.PRNGKey(seed)
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
        _pre = self.comp['pre'] ## pre-synaptic spike
        _x_pre = self.comp['x_pre'] ## pre-synaptic trace
        _post = self.comp['post'] ## post-synaptic spike
        _x_post = self.comp['x_post'] ## post-synaptic trace

        if self.exp_beta is not None: ## exponential weight-depend STDP
            self.W = _evolve_expSTDP(_pre, _x_pre, _post, _x_post, self.W,
                                     self.w_bound, self.eta, self.x_tar,
                                     exp_beta=self.exp_beta, Aplus=self.Aplus,
                                     Aminus=self.Aminus, w_norm=self.w_norm)
        else: ## power law weight-depend STDP
            self.W = _evolve_powerLawSTDP(_pre, _x_pre, _post, _x_post, self.W,
                                          self.w_bound, self.eta, self.x_tar,
                                          mu=self.mu, Aplus=self.Aplus,
                                          Aminus=self.Aminus, w_norm=self.w_norm)

    def custom_dump(self, node_directory, template=False) -> dict[str, any]:
        if not template:
            jnp.save(node_directory + "/W.npy", self.W)
        required_keys = ['shape', 'mu', 'exp_beta', 'x_tar', 'Aplus',
                         'Aminus', 'sign', 'eta', 'w_norm']
        return {k: self.__dict__.get(k, None) for k in required_keys}

    def custom_load(self, node_directory):
        if os.path.isfile(node_directory + "/W.npy"):
            self.W = jnp.load(node_directory + "/W.npy")

class_name = TrSTDPSynapse.__name__
