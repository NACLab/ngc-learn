from jax import random, numpy as jnp, jit
from ngclearn import resolver, Component, Compartment
from ngclearn.components.synapses import DenseSynapse
from ngclearn.utils import tensorstats

def _calc_update(dt, pre, x_pre, post, x_post, W, w_bound=1., x_tar=0.7,
                 exp_beta=1., Aplus=1., Aminus=0.): ## internal dynamics method
    ## equations 4 from Diehl and Cook - full exponential weight-dependent STDP
    ## calculate post-synaptic term
    post_term1 = jnp.exp(-exp_beta * W) * jnp.matmul(x_pre.T, post)
    x_tar_vec = x_pre * 0 + x_tar  # need to broadcast scalar x_tar to mat/vec form
    post_term2 = jnp.exp(-exp_beta * (w_bound - W)) * jnp.matmul(x_tar_vec.T,
                                                                 post)
    dWpost = (post_term1 - post_term2) * Aplus
    ## calculate pre-synaptic term
    dWpre = 0.
    if Aminus > 0.:
        dWpre = -jnp.exp(-exp_beta * W) * jnp.matmul(pre.T, x_post) * Aminus
    ## calc final weighted adjustment
    dW = (dWpost + dWpre)
    return dW

class ExpSTDPSynapse(DenseSynapse):
    """
    A synaptic cable that adjusts its efficacies via trace-based form of
    spike-timing-dependent plasticity (STDP) based on an exponential weight
    dependence (the strength of which is controlled by a factor).

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

    | References:
    | Nessler, Bernhard, et al. "Bayesian computation emerges in generic cortical
    | microcircuits through spike-timing-dependent plasticity." PLoS computational
    | biology 9.4 (2013): e1003037.
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

        exp_beta: controls effect of exponential Hebbian shift/dependency

        eta: global learning rate

        pretrace_target: controls degree of pre-synaptic disconnect, i.e., amount of decay
                 (higher -> lower synaptic values)

        weight_init: a kernel to drive initialization of this synaptic cable's values;
            typically a tuple with 1st element as a string calling the name of
            initialization to use

        resist_scale: a fixed scaling (resistance) factor to apply to synaptic transform
            (Default: 1.), i.e., yields: out = ((W * Rscale) * in) + b

        p_conn: probability of a connection existing (default: 1.); setting
            this to < 1. will result in a sparser synaptic structure

        w_bound: maximum value/magnitude any synaptic efficacy can be (default: 1)
    """

    # Define Functions
    def __init__(self, name, shape, A_plus, A_minus, exp_beta, eta=1.,
                 pretrace_target=0., weight_init=None, resist_scale=1.,
                 p_conn=1., w_bound=1., batch_size=1, **kwargs):
        super().__init__(name, shape, weight_init, None, resist_scale,
                         p_conn, batch_size=batch_size, **kwargs)

        ## Exp-STDP meta-parameters
        self.shape = shape ## shape of synaptic efficacy matrix
        self.eta = eta ## global learning rate governing plasticity
        self.exp_beta = exp_beta ## if not None, will trigger exp-depend STPD rule
        self.preTrace_target = pretrace_target ## target (pre-synaptic) trace activity value # 0.7
        self.Aplus = A_plus ## LTP strength
        self.Aminus = A_minus ## LTD strength
        self.Rscale = resist_scale ## post-transformation scale factor
        self.w_bound = w_bound #1. ## soft weight constraint

        ## Compartment setup
        preVals = jnp.zeros((self.batch_size, shape[0]))
        postVals = jnp.zeros((self.batch_size, shape[1]))
        self.preSpike = Compartment(preVals)
        self.postSpike = Compartment(postVals)
        self.preTrace = Compartment(preVals)
        self.postTrace = Compartment(postVals)
        self.dWeights = Compartment(self.weights.value * 0)
        self.eta = Compartment(jnp.ones((1, 1)) * eta) ## global learning rate governing plasticity

    @staticmethod
    def _compute_update(dt, w_bound, preTrace_target, exp_beta, Aplus, Aminus,
                        preSpike, postSpike, preTrace, postTrace, weights):
        dW = _calc_update(dt, preSpike, preTrace, postSpike, postTrace, weights,
                          w_bound=w_bound, x_tar=preTrace_target, exp_beta=exp_beta,
                          Aplus=Aplus, Aminus=Aminus)
        return dW

    @staticmethod
    def _evolve(dt, w_bound, preTrace_target, exp_beta, Aplus, Aminus,
                preSpike, postSpike, preTrace, postTrace, weights, eta):
        dW = ExpSTDPSynapse._compute_update(
            dt, w_bound, preTrace_target, exp_beta, Aplus, Aminus,
            preSpike, postSpike, preTrace, postTrace, weights
        )
        ## do a gradient ascent update/shift
        _W = weights + dW * eta
        ## enforce non-negativity
        eps = 0.01
        _W = jnp.clip(_W, eps, w_bound - eps)
        return weights, dW

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
        preTrace = preVals
        postTrace = postVals
        dWeights = jnp.zeros(shape)
        return inputs, outputs, preSpike, postSpike, preTrace, postTrace, dWeights

    @resolver(_reset)
    def reset(self, inputs, outputs, preSpike, postSpike, preTrace, postTrace, dWeights):
        self.inputs.set(inputs)
        self.outputs.set(outputs)
        self.preSpike.set(preSpike)
        self.postSpike.set(postSpike)
        self.preTrace.set(preTrace)
        self.postTrace.set(postTrace)
        self.dWeights.set(dWeights)

    @classmethod
    def help(cls): ## component help function
        properties = {
            "synapse_type": "ExpSTDPSynapse - performs an adaptable synaptic "
                            "transformation of inputs to produce output signals; "
                            "synapses are adjusted with exponential trace-based "
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
            "exp_beta": "Controls effect of exponential Hebbian shift / dependency (B)",
            "eta": "Global learning rate initial condition",
            "pretrace_target": "Pre-synaptic disconnecting/decay factor (x_tar)",
        }
        info = {cls.__name__: properties,
                "compartments": compartment_props,
                "dynamics": "outputs = [(W * Rscale) * inputs] ;"
                            "dW_{ij}/dt = A_plus * [z_j * exp(-B w) - x_tar * exp(-B(w_max - w))] * s_i -"
                            "A_minus * s_j * [z_i * exp(-B w)]",
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

if __name__ == '__main__':
    from ngcsimlib.context import Context
    with Context("Bar") as bar:
        Wab = ExpSTDPSynapse("Wab", (2, 3), 1, 1, 1, 0.0004, 1)
    print(Wab)
