from jax import random, numpy as jnp, jit
from ngclearn import compilable #from ngcsimlib.parser import compilable
from ngclearn import Compartment #from ngcsimlib.compartment import Compartment
from ngclearn.components.synapses.denseSynapse import DenseSynapse

class EventSTDPSynapse(DenseSynapse): # event-driven, post-synaptic STDP
    """
    A synaptic cable that adjusts its efficacies via event-driven, post-synaptic
    spike-timing-dependent plasticity (STDP).

    | --- Synapse Compartments: ---
    | inputs - input (takes in external signals)
    | outputs - output signals (transformation induced by synapses)
    | weights - current value matrix of synaptic efficacies
    | key - JAX PRNG key
    | --- Synaptic Plasticity Compartments: ---
    | pre_tols - pre-synaptic time-of-last-spike (tols) to drive 1st term of STDP update (takes in external values)
    | postSpike - post-synaptic spike to drive 2nd term of STDP update (takes in external signals)
    | dWeights - current delta matrix containing changes to be applied to synaptic efficacies

    | References:
    | Tavanaei, Amirhossein, Timothée Masquelier, and Anthony Maida.
    | "Representation learning using event-based STDP." Neural Networks 105
    | (2018): 294-303.

    Args:
        name: the string name of this cell

        shape: tuple specifying shape of this synaptic cable (usually a 2-tuple
            with number of inputs by number of outputs)

        eta: global learning rate initial condition

        lmbda: controls degree of synaptic disconnect ("lambda")

        A_plus: strength of long-term potentiation (LTP)

        A_minus: strength of long-term depression (LTD)

        presyn_win_len: pre-synaptic window time, or how far back in time to
            look for the presence of a pre-synaptic spike, in milliseconds (default: 1 ms)

        w_bound: maximum value/magnitude any synaptic efficacy can be (default: 1)

        weight_init: a kernel to drive initialization of this synaptic cable's values;
            typically a tuple with 1st element as a string calling the name of
            initialization to use

        resist_scale: a fixed scaling factor to apply to synaptic transform
            (Default: 1), i.e., yields: out = ((W * Rscale) * in) + b

        p_conn: probability of a connection existing (default: 1.); setting
            this to < 1. will result in a sparser synaptic structure
    """

    def __init__(
            self, name, shape, eta, lmbda=0.01, A_plus=1., A_minus=1., presyn_win_len=2., w_bound=1., 
            weight_init=None, resist_scale=1., p_conn=1., batch_size=1, **kwargs
    ):
        super().__init__(name, shape, weight_init, None, resist_scale, p_conn, batch_size=batch_size, **kwargs)

        ## Synaptic hyper-parameters
        self.eta = eta ## global learning rate governing plasticity
        self.lmbda = lmbda ## controls scaling of STDP rule
        self.presyn_win_len = presyn_win_len
        assert self.presyn_win_len >= 0. ## pre-synaptic window must be non-negative
        self.Aplus = A_plus
        self.Aminus = A_minus
        self.Rscale = resist_scale ## post-transformation scale factor
        self.w_bound = w_bound ## soft weight constraint

        ## Compartment setup
        preVals = jnp.zeros((self.batch_size, shape[0]))
        postVals = jnp.zeros((self.batch_size, shape[1]))
        self.pre_tols = Compartment(preVals)
        self.postSpike = Compartment(postVals)
        self.dWeights = Compartment(self.weights.get() * 0)
        self.eta = Compartment(jnp.ones((1, 1)) * eta)  ## global learning rate governing plasticity

    def _compute_update(self, t, dt): ## synaptic adjustment calculation co-routine
        ## check if a spike occurred in window of (t - presyn_win_len, t]
        m = (self.pre_tols.get() > 0.) * 1.  ## ignore default value of tols = 0 ms
        if self.presyn_win_len > 0.:
            lbound = ((t - self.presyn_win_len) < self.pre_tols.get()) * 1.
            preSpike = lbound * m
        else:
            check_spike = (self.pre_tols.get() == t) * 1.
            preSpike = check_spike * m
        ## this implements a generalization of the rule in eqn 18 of the paper
        pos_shift = self.w_bound - (self.weights.get() * (1. + self.lmbda))
        pos_shift = pos_shift * self.Aplus
        neg_shift = -self.weights.get() * (1. + self.lmbda)
        neg_shift = neg_shift * self.Aminus
        dW = jnp.where(preSpike.T, pos_shift, neg_shift) # at pre-spikes => LTP, else decay
        dW = (dW * self.postSpike.get()) ## gate to make sure only post-spikes trigger updates
        return dW

    @compilable
    def evolve(self, t, dt):
        dWeights = self._compute_update(t, dt)
        weights = self.weights.get() + dWeights * self.eta  # * (1. - w) * eta
        weights = jnp.clip(weights, 0.01, self.w_bound)  ## Note: this step not in source paper

        self.weights.set(weights)
        self.dWeights.set(dWeights)

    @compilable
    def reset(self):
        preVals = jnp.zeros((self.batch_size.get(), self.shape.get()[0]))
        postVals = jnp.zeros((self.batch_size.get(), self.shape.get()[1]))

        if not self.inputs.targeted:
            self.inputs.set(preVals)
        self.outputs.set(postVals)
        self.pre_tols.set(preVals)  ## pre-synaptic time-of-last-spike(s) record
        self.postSpike.set(postVals)
        self.dWeights.set(jnp.zeros(self.shape.get()))

    @classmethod
    def help(cls): ## component help function
        properties = {
            "synapse_type": "EventSTDPSynapse - performs an adaptable synaptic "
                            "transformation of inputs to produce output signals; "
                            "synapses are adjusted with event-based post-synaptic "
                            "spike-timing-dependent plasticity (STDP)"
        }
        compartment_props = {
            "inputs":
                {"inputs": "Takes in external input signal values",
                 "pre_tols": "Pre-synaptic time-of-last-spike (`tols` for s_j)",
                 "postSpike": "Post-synaptic spike compartment value/term for STDP (s_i)"},
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
            "lmbda": "Degree of synaptic disconnect",
            "eta": "Global learning rate (multiplier beyond A_plus and A_minus)",
            "w_bound ": "Maximum value/magnitude that any single synapse can take"
        }
        info = {cls.__name__: properties,
                "compartments": compartment_props,
                "dynamics": "outputs = [(W * Rscale) * inputs] ;"
                            "dW_{ij}/dt = eta * [ (1 - W_{ij}(1 + lmbda)) * s_j - W_{ij} * (1 + lmbda) * s_j ]",
                "hyperparameters": hyperparams}
        return info

if __name__ == '__main__':
    from ngcsimlib.context import Context
    with Context("Bar") as bar:
        Wab = EventSTDPSynapse("Wab", (2, 3), 1.)
    print(Wab)
