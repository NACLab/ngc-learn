from jax import numpy as jnp, jit
from ngclearn import resolver, Component, Compartment
from ngclearn.utils import tensorstats
## import parent synapse class/component
from ngclearn.components.synapses import DenseSynapse

def evolve(pre, post, W, eta=0.00005, lmbda=0., w_bound=1.):
    """
    Evolves/changes the synpatic value matrix underlying this synaptic cable,
    given relevant statistics.

    Args:
        pre: pre-synaptic statistic to drive update

        post: post-synaptic statistic to drive update

        W: synaptic weight values (at time t)

        w_bound: maximum value to enforce over newly computed efficacies

        eta: global learning rate to apply to the Hebbian update

        lmbda: synaptic change control coefficient ("lambda")

    Returns:
        the newly evolved synaptic weight value matrix, synaptic update matrix
    """
    pos_shift = w_bound - W * (1. + lmbda) # this follows rule in eqn 18 of the paper
    neg_shift = -W * (1. + lmbda)
    dW = jnp.where(pre.T, pos_shift, neg_shift)
    dW = (dW * post)
    W = W + dW * eta #* (1. - w) * eta
    W = jnp.clip(W, 0.01, w_bound) # not in source paper
    return W, dW

class EventSTDPSynapse(DenseSynapse): # event-driven, post-synaptic STDP
    """
    A synaptic cable that adjusts its efficacies via event-driven, post-synaptic
    spike-timing-dependent plasticity (STDP).

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
    | Tavanaei, Amirhossein, Timothée Masquelier, and Anthony Maida.
    | "Representation learning using event-based STDP." Neural Networks 105
    | (2018): 294-303.

    Args:
        name: the string name of this cell

        shape: tuple specifying shape of this synaptic cable (usually a 2-tuple
            with number of inputs by number of outputs)

        eta: global learning rate

        lmbda: controls degree of synaptic disconnect ("lambda")

        weight_init: a kernel to drive initialization of this synaptic cable's values;
            typically a tuple with 1st element as a string calling the name of
            initialization to use

        resist_scale: a fixed scaling factor to apply to synaptic transform
            (Default: 1.), i.e., yields: out = ((W * Rscale) * in) + b

        p_conn: probability of a connection existing (default: 1.); setting
            this to < 1. will result in a sparser synaptic structure
    """

    # Define Functions
    def __init__(self, name, shape, eta, lmbda=0.01, w_bound=1.,
                 weight_init=None, resist_scale=1., p_conn=1., **kwargs):
        super().__init__(name, shape, weight_init, None, resist_scale, p_conn, **kwargs)

        ## Synaptic hyper-parameters
        self.eta = eta ## global learning rate governing plasticity
        self.lmbda = lmbda ## controls scaling of STDP rule
        self.Rscale = resist_scale ## post-transformation scale factor
        self.w_bound = w_bound ## soft weight constraint

        ## Compartment setup
        preVals = jnp.zeros((self.batch_size, shape[0]))
        postVals = jnp.zeros((self.batch_size, shape[1]))
        self.preSpike = Compartment(preVals)
        self.postSpike = Compartment(postVals)
        self.dWeights = Compartment(self.weights.value * 0)

    @staticmethod
    def _evolve(t, dt, eta, lmbda, w_bound, preSpike, postSpike, weights):
        weights, dWeights = evolve(preSpike, postSpike, weights, eta, lmbda, w_bound)
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
        dWeights = jnp.zeros(shape)
        return inputs, outputs, preSpike, postSpike, dWeights

    @resolver(_reset)
    def reset(self, inputs, outputs, preSpike, postSpike, dWeights):
        self.inputs.set(inputs)
        self.outputs.set(outputs)
        self.preSpike.set(preSpike)
        self.postSpike.set(postSpike)
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
        Wab = EventSTDPSynapse("Wab", (2, 3), 1.)
    print(Wab)
