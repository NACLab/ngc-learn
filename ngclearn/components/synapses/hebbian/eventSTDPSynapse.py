from jax import random, numpy as jnp, jit
from ngclearn import resolver, Component, Compartment
from ngclearn.utils import tensorstats
from ngclearn.utils.model_utils import initialize_params
import time

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

@jit
def compute_layer(inp, weight, scale=1.):
    """
    Applies the transformation/projection induced by the synaptic efficacie
    associated with this synaptic cable

    Args:
        inp: signal input to run through this synaptic cable

        weight: this cable's synaptic value matrix

        scale: scale factor to apply to synapses before transform applied
            to input values

    Returns:
        a projection/transformation of input "inp"
    """
    return jnp.matmul(inp, weight * scale)

class EventSTDPSynapse(Component): # event-driven, post-synaptic STDP
    """
    A synaptic cable that adjusts its efficacies via event-driven, post-synaptic
    spike-timing-dependent plasticity (STDP).

    | --- Synapse Compartments: ---
    | inputs - input (takes in external signals)
    | outputs - output signal (transformation induced by synapses)
    | weights - current value matrix of synaptic efficacies
    | --- Synaptic Plasticity Compartments: ---
    | preSpike - pre-synaptic spike to drive 1st term of STDP update
    | postSpike - post-synaptic spike to drive 2nd term of STDP update
    | preTrace - pre-synaptic trace value to drive 1st term of STDP update
    | postTrace - post-synaptic trace value to drive 2nd term of STDP update
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

        wInit: a kernel to drive initialization of this synaptic cable's values;
            typically a tuple with 1st element as a string calling the name of
            initialization to use, e.g., ("uniform", -0.1, 0.1) samples U(-1,1)
            for each dimension/value of this cable's underlying value matrix

        Rscale: a fixed scaling factor to apply to synaptic transform
            (Default: 1.), i.e., yields: out = ((W * Rscale) * in) + b

        key: PRNG key to control determinism of any underlying random values
            associated with this synaptic cable

        directory: string indicating directory on disk to save synaptic parameter
            values to (i.e., initial threshold values and any persistent adaptive
            threshold values)
    """

    # Define Functions
    def __init__(self, name, shape, eta, lmbda=0.01, w_bound=1.,
                 wInit=("uniform", 0.025, 0.8), Rscale=1., key=None,
                 directory=None, **kwargs):
        super().__init__(name, **kwargs)

        ## constructor-only rng setup
        tmp_key = random.PRNGKey(time.time_ns()) if key is None else key

        ##parms
        self.shape = shape ## shape of synaptic efficacy matrix
        self.eta = eta ## global learning rate governing plasticity
        self.lmbda = lmbda ## controls scaling of STDP rule
        self.shape = shape  ## shape of synaptic matrix W
        self.Rscale = Rscale ## post-transformation scale factor
        self.w_bound = w_bound ## soft weight constraint

        tmp_key, subkey = random.split(tmp_key)
        weights = initialize_params(subkey, wInit, shape)

        self.batch_size = 1
        ## Compartment setup
        preVals = jnp.zeros((self.batch_size, shape[0]))
        postVals = jnp.zeros((self.batch_size, shape[1]))
        self.inputs = Compartment(preVals)
        self.outputs = Compartment(postVals)
        self.preSpike = Compartment(preVals)
        self.postSpike = Compartment(postVals)
        self.weights = Compartment(weights)
        self.dWeights = Compartment(weights * 0)
        #self.reset()

    @staticmethod
    def _advance_state(t, dt, Rscale, inputs, weights):
        ## run signals across synapses
        outputs = compute_layer(inputs, weights, Rscale)
        return outputs

    @resolver(_advance_state)
    def advance_state(self, outputs):
        self.outputs.set(outputs)

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

    def save(self, directory, **kwargs):
        file_name = directory + "/" + self.name + ".npz"
        jnp.savez(file_name, weights=self.weights.value)

    def load(self, directory, **kwargs):
        file_name = directory + "/" + self.name + ".npz"
        data = jnp.load(file_name)
        self.weights.set( data['weights'] )

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
