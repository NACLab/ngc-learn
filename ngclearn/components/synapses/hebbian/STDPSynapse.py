from jax import random, numpy as jnp, jit
from ngclearn import resolver, Component, Compartment
from ngclearn.components.synapses import DenseSynapse
from ngclearn.utils import tensorstats

class STDPSynapse(DenseSynapse): # power-law / trace-based STDP
    """
    A synaptic cable that adjusts its efficacies via raw
    spike-timing-dependent plasticity (STDP).

    | --- Synapse Compartments: ---
    | inputs - input (takes in external signals)
    | outputs - output signals (transformation induced by synapses)
    | weights - current value matrix of synaptic efficacies
    | key - JAX PRNG key
    | --- Synaptic Plasticity Compartments: ---
    | preSpike - pre-synaptic spike to drive long-term potentiation (takes in external signals)
    | postSpike - post-synaptic spike to drive long-term depression (takes in external signals)
    | pre_tols - pre-synaptic time-of-last-spike (takes in external signals)
    | post_tols - post-synaptic time-of-last-spike (takes in external signals)
    | dWeights - current delta matrix containing changes to be applied to synaptic efficacies
    | eta - global learning rate (multiplier beyond A_plus and A_minus)

    | References:
    | Markram, Henry, et al. "Regulation of synaptic efficacy by coincidence of
    | postsynaptic APs and EPSPs." Science 275.5297 (1997): 213-215.
    |
    | Bi, Guo-qiang, and Mu-ming Poo. "Synaptic modification by correlated
    | activity: Hebb's postulate revisited." Annual review of neuroscience 24.1
    | (2001): 139-166.

    Args:
        name: the string name of this cell

        shape: tuple specifying shape of this synaptic cable (usually a 2-tuple
            with number of inputs by number of outputs)

        A_plus: strength of long-term potentiation (LTP)

        A_minus: strength of long-term depression (LTD)

        tau_plus: time constant of long-term potentiation (LTP)

        tau_minus: time constant of long-term depression (LTD)

        eta: global learning rate initial value/condition (default: 1)

        tau_w: time constant for synaptic adjustment; setting this to zero
            disables Euler-style synaptic adjustment (default: 0)

        weight_init: a kernel to drive initialization of this synaptic cable's values;
            typically a tuple with 1st element as a string calling the name of
            initialization to use

        resist_scale: a fixed scaling factor to apply to synaptic transform
            (Default: 1.), i.e., yields: out = ((W * Rscale) * in)

        p_conn: probability of a connection existing (default: 1); setting
            this to < 1. will result in a sparser synaptic structure

        w_bound: maximum value/magnitude any synaptic efficacy can be (default: 1)
    """

    # Define Functions
    def __init__(self, name, shape, A_plus, A_minus, tau_plus=10., tau_minus=10., w_decay=0., 
                 eta=1., tau_w=0., weight_init=None, resist_scale=1., p_conn=1., w_bound=1.,
                 batch_size=1, **kwargs):
        super().__init__(name, shape, weight_init, None, resist_scale,
                         p_conn, batch_size=batch_size, **kwargs)
        assert self.batch_size == 1 ## note: STDP only supports online learning in this implementation
        ## Synaptic hyper-parameters
        self.shape = shape ## shape of synaptic efficacy matrix
        self.Aplus = A_plus ## LTP strength
        self.Aminus = A_minus ## LTD strength
        self.tau_plus = tau_plus ## LTP time constant
        self.tau_minus = tau_minus ## LTD time constant
        self.Rscale = resist_scale ## post-transformation scale factor
        self.w_bound = w_bound #1. ## soft weight constraint
        self.tau_w = tau_w ## synaptic update time constant
        self.w_decay = w_decay

        ## Compartment setup
        preVals = jnp.zeros((self.batch_size, shape[0]))
        postVals = jnp.zeros((self.batch_size, shape[1]))
        self.preSpike = Compartment(preVals)
        self.postSpike = Compartment(postVals)
        self.pre_tols = Compartment(preVals) ## pre-synaptic time-of-last-spike
        self.post_tols = Compartment(postVals) ## post-synaptic time-of-last-spike
        self.dWeights = Compartment(self.weights.value * 0)
        self.eta = Compartment(jnp.ones((1, 1)) * eta) ## global learning rate

    @staticmethod
    def _compute_update(Aplus, Aminus, tau_plus, tau_minus, preSpike, postSpike,
                        pre_tols, post_tols, weights):
        ## calculate time deltas matrix block --> (t_post - t_pre)
        post_m = (post_tols > 0.) ## zero post-tols mask
        pre_m = (pre_tols > 0.).T ## zero pre-tols mask
        t_delta = ((weights * 0 + 1.) * post_tols) - pre_tols.T ## t_delta.shape = weights.shape
        t_delta = t_delta * post_m * pre_m  ## mask out zero tols and same-time spikes
        pos_t_delta_m = (t_delta > 0.) ## positive t-delta mask
        neg_t_delta_m = (t_delta < 0.) ## negative t-delta mask
        #t_delta = t_delta * pos_t_delta_m + t_delta * neg_t_delta_m ## mask out same time spikes
        ## calculate post-synaptic term
        postTerm = jnp.exp(-t_delta/tau_plus) * pos_t_delta_m
        dWpost = postTerm * (postSpike * Aplus)
        dWpre = 0.
        if Aminus > 0.:
            ## calculate pre-synaptic term
            preTerm = jnp.exp(-t_delta / tau_minus) * neg_t_delta_m
            dWpre = -preTerm * (preSpike.T * Aminus)
        ## calc final weighted adjustment
        dW = (dWpost + dWpre)
        return dW

    @staticmethod
    def _evolve(dt, w_bound, w_decay, tau_w, Aplus, Aminus, tau_plus, tau_minus, preSpike,
                postSpike, pre_tols, post_tols, weights, eta):
        dWeights = STDPSynapse._compute_update(
            Aplus, Aminus, tau_plus, tau_minus, preSpike, postSpike, pre_tols,
            post_tols, weights
        )
        ## shift/alter values of synaptic efficacies
        if tau_w > 0.: ## triggers Euler-style synaptic update
            weights = weights + (-weights * dt/tau_w + dWeights * eta)
        else: ## raw simple ascent-style update
            weights = weights + dWeights * eta - weights * w_decay
        ## enforce non-negativity
        eps = 0.001 # 0.01
        weights = jnp.clip(weights, eps, w_bound - eps)  # jnp.abs(w_bound))
        return weights, dWeights

    @resolver(_evolve)
    def evolve(self, weights, dWeights):
        self.weights.set(weights)
        self.dWeights.set(dWeights)

    @staticmethod
    def _reset(batch_size, shape):
        preVals = jnp.zeros((batch_size, shape[0]))
        postVals = jnp.zeros((batch_size, shape[1]))
        inputs = preVals
        outputs = postVals
        preSpike = preVals
        postSpike = postVals
        pre_tols = preVals
        post_tols = postVals
        dWeights = jnp.zeros(shape)
        return inputs, outputs, preSpike, postSpike, pre_tols, post_tols, dWeights

    @resolver(_reset)
    def reset(self, inputs, outputs, preSpike, postSpike, pre_tols, post_tols, dWeights):
        self.inputs.set(inputs)
        self.outputs.set(outputs)
        self.preSpike.set(preSpike)
        self.postSpike.set(postSpike)
        self.pre_tols.set(pre_tols)
        self.post_tols.set(post_tols)
        self.dWeights.set(dWeights)

    @classmethod
    def help(cls): ## component help function
        properties = {
            "synapse_type": "STDPSynapse - performs an adaptable synaptic "
                            "transformation of inputs to produce output signals; "
                            "synapses are adjusted with classical "
                            "spike-timing-dependent plasticity (STDP)"
        }
        compartment_props = {
            "inputs":
                {"inputs": "Takes in external input signal values",
                 "preSpike": "Pre-synaptic spike compartment event for STDP (s_j)",
                 "postSpike": "Post-synaptic spike compartment event for STDP (s_i)",
                 "pre_tols": "Pre-synaptic time-of-last-spike (t_j)",
                 "post_tols": "Post-synaptic time-of-last-spike (t_i)"},
            "states":
                {"weights": "Synapse efficacy/strength parameter values",
                 "biases": "Base-rate/bias parameter values",
                 "eta": "Global learning rate (multiplier beyond A_plus and A_minus)",
                 "key": "JAX PRNG key"},
            "analytics":
                {"dWeights": "Synaptic weight value adjustment matrix produced at time t"},
            "outputs":
                {"outputs": "Output of synaptic transformation"},
        }
        hyperparams = {
            "shape": "Shape of synaptic weight value matrix; number inputs x number outputs",
            "batch_size": "Batch size dimension of this component",
            "weight_init": "Initialization conditions for synaptic weight (W) values",
            "resist_scale": "Resistance level scaling factor (applied to output of transformation)",
            "p_conn": "Probability of a connection existing (otherwise, it is masked to zero)",
            "A_plus": "Strength of long-term potentiation (LTP)",
            "A_minus": "Strength of long-term depression (LTD)",
            "tau_plus": "Time constant for long-term potentiation (LTP)",
            "tau_minus": "Time constant for long-term depression (LTD)",
            "eta": "Global learning rate initial condition",
            "tau_w": "Time constant for synaptic adjustment (if Euler-style change used)"
        }
        info = {cls.__name__: properties,
                "compartments": compartment_props,
                "dynamics": "outputs = [(W * Rscale) * inputs] ;"
                            "dW_{ij}/dt = A_plus * exp(-(t_i - t_j)/tau_plus) * s_j -"
                            " A_minus exp(-(t_i - t_j)/tau_minus) * s_i",
                "hyperparameters": hyperparams}
        return info

    def __repr__(self):
        comps = [varname for varname in dir(self) if Compartment.is_compartment(getattr(self, varname))]
        maxlen = max(len(c) for c in comps) + 5
        lines = f"[{self.__class__.__name__}] PATH: {self.name}\n"
        for c in comps:
            stats = tensorstats(getattr(self, c).value)
            if stats is not None:
                line = [f"{k}: {v}" for k, v in stats.items()]
                line = ", ".join(line)
            else:
                line = "None"
            lines += f"  {f'({c})'.ljust(maxlen)}{line}\n"
        return lines
