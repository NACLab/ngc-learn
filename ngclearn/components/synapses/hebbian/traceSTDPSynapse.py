from jax import random, numpy as jnp, jit
from ngclearn import resolver, Component, Compartment
from ngclearn.components.synapses import DenseSynapse
from ngclearn.utils import tensorstats

def evolve(dt, pre, x_pre, post, x_post, W, w_bound=1., eta=1., x_tar=0.0,
           mu=0., Aplus=1., Aminus=0.):
    """
    Evolves/changes the synpatic value matrix underlying this synaptic cable,
    given relevant statistics.

    Args:
        pre: pre-synaptic statistic to drive update

        x_pre: pre-synaptic trace value

        post: post-synaptic statistic to drive update

        x_post: post-synaptic trace value

        W: synaptic weight values (at time t)

        w_bound: maximum value to enforce over newly computed efficacies

        eta: global learning rate to apply to the Hebbian update

        x_tar: controls degree of pre-synaptic disconnect

        mu: controls the power scale of the Hebbian shift

        Aplus: strength of long-term potentiation (LTP)

        Aminus: strength of long-term depression (LTD)

    Returns:
        the newly evolved synaptic weight value matrix, synaptic update matrix
    """
    if mu > 0.:
        ## equations 3, 5, & 6 from Diehl and Cook - full power-law STDP
        post_shift = jnp.power(w_bound - W, mu)
        pre_shift = jnp.power(W, mu)
        dWpost = (post_shift * jnp.matmul((x_pre - x_tar).T, post)) * Aplus
        dWpre = 0.
        if Aminus > 0.:
            dWpre = -(pre_shift * jnp.matmul(pre.T, x_post)) * Aminus
    else:
        ## calculate post-synaptic term
        dWpost = jnp.matmul((x_pre - x_tar).T, post * Aplus)
        dWpre = 0.
        if Aminus > 0.:
            ## calculate pre-synaptic term
            dWpre = -jnp.matmul(pre.T, x_post * Aminus)
    ## calc final weighted adjustment
    dW = (dWpost + dWpre)
    ## do a gradient ascent update/shift
    _W = W + dW * eta
    ## enforce non-negativity
    eps = 0.01 # 0.001
    _W = jnp.clip(_W, eps, w_bound - eps) #jnp.abs(w_bound)) # 0.01, w_bound)
    #print(_W)
    return _W, dW
class TraceSTDPSynapse(DenseSynapse): # power-law / trace-based STDP
    """
    A synaptic cable that adjusts its efficacies via trace-based form of
    spike-timing-dependent plasticity (STDP), including an optional power-scale
    dependence that can be equipped to the Hebbian adjustment (the strength of
    which is controlled by a scalar factor).

    | --- Synapse Compartments: ---
    | inputs - input (takes in external signals)
    | outputs - output signal (transformation induced by synapses)
    | weights - current value matrix of synaptic efficacies
    | key - JAX RNG key
    | --- Synaptic Plasticity Compartments: ---
    | preSpike - pre-synaptic spike to drive 1st term of STDP update (takes in external signals)
    | postSpike - post-synaptic spike to drive 2nd term of STDP update (takes in external signals)
    | preTrace - pre-synaptic trace value to drive 1st term of STDP update (takes in external signals)
    | postTrace - post-synaptic trace value to drive 2nd term of STDP update (takes in external signals)
    | dWeights - current delta matrix containing changes to be applied to synaptic efficacies

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

        eta: global learning rate

        mu: controls the power scale of the Hebbian shift

        pretrace_target: controls degree of pre-synaptic disconnect, i.e., amount of decay
                 (higher -> lower synaptic values)

        weight_init: a kernel to drive initialization of this synaptic cable's values;
            typically a tuple with 1st element as a string calling the name of
            initialization to use

        resist_scale: a fixed scaling factor to apply to synaptic transform
            (Default: 1.), i.e., yields: out = ((W * Rscale) * in)

        p_conn: probability of a connection existing (default: 1.); setting
            this to < 1. will result in a sparser synaptic structure
    """

    # Define Functions
    def __init__(self, name, shape, A_plus, A_minus, eta=1., mu=0.,
                 pretrace_target=0., weight_init=None, resist_scale=1.,
                 p_conn=1., **kwargs):
        super().__init__(name, shape, weight_init, None, resist_scale,
                         p_conn, **kwargs)

        ## Synaptic hyper-parameters
        self.shape = shape ## shape of synaptic efficacy matrix
        self.eta = eta ## global learning rate governing plasticity
        self.mu = mu ## controls power-scaling of STDP rule
        self.preTrace_target = pretrace_target ## target (pre-synaptic) trace activity value # 0.7
        self.Aplus = A_plus ## LTP strength
        self.Aminus = A_minus ## LTD strength
        self.Rscale = resist_scale ## post-transformation scale factor
        self.w_bound = 1. ## soft weight constraint

        self.batch_size = 1
        ## Compartment setup
        preVals = jnp.zeros((self.batch_size, shape[0]))
        postVals = jnp.zeros((self.batch_size, shape[1]))
        self.preSpike = Compartment(preVals)
        self.postSpike = Compartment(postVals)
        self.preTrace = Compartment(preVals)
        self.postTrace = Compartment(postVals)
        self.dWeights = Compartment(self.weights.value * 0)

    @staticmethod
    def _evolve(t, dt, w_bound, eta, preTrace_target, mu, Aplus, Aminus,
                preSpike, postSpike, preTrace, postTrace, weights):
        weights, dW = evolve(dt, preSpike, preTrace, postSpike, postTrace, weights,
                             w_bound=w_bound, eta=eta, x_tar=preTrace_target, mu=mu,
                             Aplus=Aplus, Aminus=Aminus)
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
        Wab = TraceSTDPSynapse("Wab", (2, 3), 1, 1, 0.0004)
    print(Wab)
