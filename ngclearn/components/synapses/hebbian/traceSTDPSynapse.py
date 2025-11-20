from jax import random, numpy as jnp, jit
from ngclearn import compilable #from ngcsimlib.parser import compilable
from ngclearn import Compartment #from ngcsimlib.compartment import Compartment
from ngclearn.components.synapses.denseSynapse import DenseSynapse


class TraceSTDPSynapse(DenseSynapse): # power-law / trace-based STDP
    """
    A synaptic cable that adjusts its efficacies via trace-based form of
    spike-timing-dependent plasticity (STDP), including an optional power-scale
    dependence that can be equipped to the Hebbian adjustment (the strength of
    which is controlled by a scalar factor).

    | --- Synapse Compartments: ---
    | inputs - input (takes in external signals)
    | outputs - output signals (transformation induced by synapses)
    | weights - current value matrix of synaptic efficacies
    | key - JAX PRNG key
    | --- Synaptic Plasticity Compartments: ---
    | preSpike - pre-synaptic spike to drive 1st term of STDP update (takes in external signals)
    | postSpike - post-synaptic spike to drive 2nd term of STDP update (takes in external signals)
    | preTrace - pre-synaptic trace value to drive 1st term of STDP update (takes in external signals)
    | postTrace - post-synaptic trace value to drive 2nd term of STDP update (takes in external signals)
    | dWeights - current delta matrix containing changes to be applied to synaptic efficacies
    | eta - global learning rate (multiplier beyond A_plus and A_minus)

    | References:
    | Morrison, Abigail, Ad Aertsen, and Markus Diesmann. "Spike-timing-dependent
    | plasticity in balanced random networks." Neural computation 19.6 (2007): 1437-1467.
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

        eta: global learning rate initial value/condition (default: 1)

        mu: controls the power scale of the Hebbian shift

        pretrace_target: controls degree of pre-synaptic disconnect, i.e., amount of decay
                 (higher -> lower synaptic values)

        weight_init: a kernel to drive initialization of this synaptic cable's values;
            typically a tuple with 1st element as a string calling the name of
            initialization to use

        resist_scale: a fixed scaling factor to apply to synaptic transform
            (Default: 1.), i.e., yields: out = ((W * resistance) * in)

        p_conn: probability of a connection existing (default: 1); setting
            this to < 1. will result in a sparser synaptic structure

        w_bound: maximum value/magnitude any synaptic efficacy can be (default: 1)

        tau_w: synaptic weight decay coefficient to apply to STDP update

        weight_mask: synaptic binary masking matrix to apply (to enforce a constant sparse structure; default: None)
    """

    def __init__(
            self, name, shape, A_plus, A_minus, eta=1., mu=0., pretrace_target=0., weight_init=None, resist_scale=1.,
            p_conn=1., w_bound=1., tau_w=0., weight_mask=None, batch_size=1, **kwargs
    ):
        super().__init__(name, shape, weight_init, None, resist_scale, p_conn, batch_size=batch_size, **kwargs)

        self.tau_w = tau_w
        self.mu = mu ## controls power-scaling of STDP rule
        self.preTrace_target = pretrace_target ## target (pre-synaptic) trace activity value # 0.7
        self.Aplus = A_plus ## LTP strength
        self.Aminus = A_minus ## LTD strength
        self.w_bound = w_bound #1. ## soft weight constraint
        self.w_eps = 0. ## w_eps = 0.01

        if weight_mask is None:
            self.weight_mask = jnp.ones((1, 1))
        else:
            self.weight_mask = weight_mask
        self.weights.set(self.weights.get() * self.weight_mask)

        ## Compartment setup
        preVals = jnp.zeros((self.batch_size, shape[0]))
        postVals = jnp.zeros((self.batch_size, shape[1]))
        self.preSpike = Compartment(preVals)
        self.postSpike = Compartment(postVals)
        self.preTrace = Compartment(preVals)
        self.postTrace = Compartment(postVals)
        self.dWeights = Compartment(self.weights.get() * 0)
        self.eta = eta ## global learning rate

    def _compute_update(self):
        if self.mu > 0.:
            post_shift = jnp.power(self.w_bound - self.weights.get(), self.mu)
            pre_shift = jnp.power(self.weights.get(), self.mu)
            dWpost = (post_shift * jnp.matmul((self.preTrace.get() - self.preTrace_target).T, self.postSpike.get())) * self.Aplus

            if self.Aminus > 0.:
                dWpre = -(pre_shift * jnp.matmul(self.preSpike.get().T, self.postTrace.get())) * self.Aminus
            else:
                dWpre = 0.

        else:
            dWpost = jnp.matmul((self.preTrace.get() - self.preTrace_target).T, self.postSpike.get() * self.Aplus)
            if self.Aminus > 0.:
                dWpre = -jnp.matmul(self.preSpike.get().T, self.postTrace.get() * self.Aminus)
            else:
                dWpre = 0.

        dW = (dWpost + dWpre)
        return dW

    @compilable
    def evolve(self):
        dWeights = self._compute_update()
        if self.tau_w > 0.:
            decayTerm = self.weights.get() / self.tau_w
        else:
            decayTerm = 0.

        # print(jnp.nonzero(dWeights))
        w = self.weights.get() + (dWeights * self.eta) - decayTerm
        w = jnp.clip(w, self.w_eps, self.w_bound - self.w_eps)
        w = jnp.where(self.weight_mask != 0., w, 0.)
        self.weights.set(w)
        self.dWeights.set(dWeights)

    @compilable
    def reset(self):
        preVals = jnp.zeros((self.batch_size.get(), self.shape.get()[0]))
        postVals = jnp.zeros((self.batch_size.get(), self.shape.get()[1]))

        if not self.inputs.targeted:
            self.inputs.set(preVals)
        self.outputs.set(postVals)
        self.preSpike.set(preVals)
        self.postSpike.set(postVals)
        self.preTrace.set(preVals)
        self.postTrace.set(postVals)
        self.dWeights.set(jnp.zeros(self.shape.get()))


    @classmethod
    def help(cls): ## component help function
        properties = {
            "synapse_type": "TraceSTDPSynapse - performs an adaptable synaptic "
                            "transformation of inputs to produce output signals; "
                            "synapses are adjusted with trace-based "
                            "spike-timing-dependent plasticity (STDP)"
        }
        compartment_props = {
            "inputs":
                {"inputs": "Takes in external input signal values",
                 "preSpike": "Pre-synaptic spike compartment value/term for STDP (s_j)",
                 "postSpike": "Post-synaptic spike compartment value/term for STDP (s_i)",
                 "preTrace": "Pre-synaptic trace value term for STDP (z_j)",
                 "postTrace": "Post-synaptic trace value term for STDP (z_i)"},
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
            "eta": "Global learning rate initial condition",
            "mu": "Power factor for STDP adjustment",
            "pretrace_target": "Pre-synaptic disconnecting/decay factor (x_tar)",
            "weight_mask" : "Binary synaptic weight mask to apply to enforce a sparsity structure"
        }
        info = {cls.__name__: properties,
                "compartments": compartment_props,
                "dynamics": "outputs = [(W * Rscale) * inputs] ;"
                            "dW_{ij}/dt = A_plus * (z_j - x_tar) * s_i - A_minus * s_j * z_i",
                "hyperparameters": hyperparams}
        return info


if __name__ == '__main__':
    from ngcsimlib.context import Context
    with Context("Bar") as bar:
        Wab = TraceSTDPSynapse("Wab", (2, 3), 1, 1, 0.0004)
    print(Wab)
