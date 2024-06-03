from jax import random, numpy as jnp, jit
from ngclearn import resolver, Component, Compartment
from ngclearn.utils import tensorstats
from ngclearn.utils.model_utils import initialize_params
import time

def evolve(dt, pre, post, theta, W, tau_w, tau_theta, w_bound=0., w_decay=0.):
    """
    Evolves/changes the synpatic value matrix and threshold variables underlying
    this synaptic cable, given relevant statistics.

    Args:
        dt: integration time constant

        pre: pre-synaptic statistic to drive update (e.g., could be a trace)

        post: post-synaptic statistic to drive update (e.g., could be a trace)

        theta: the current state of the synaptic threshold variables

        W: synaptic weight values (at time t)

        tau_w: synaptic update time constant

        tau_theta: threshold variable evolution time constant

        w_bound: maximum value to enforce over newly computed efficacies
            (default: 0.); must > 0. to be used

        w_decay: synaptic decay factor (default: 0.)

    Returns:
        the newly evolved synaptic weight value matrix,
        the newly evolved synaptic threshold variables,
        the synaptic update matrix
    """
    post_term = post - theta
    dW = jnp.matmul(pre.T, post_term)
    if w_bound > 0.:
        dW = dW * (w_bound - jnp.abs(W))
    _W = W + (-W * w_decay + dW) * dt/tau_w
    _theta = theta + (-theta + jnp.square(post)) * dt/tau_theta
    return _W, _theta, dW

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

class BCMSynapse(Component): # BCM-adjusted synaptic cable
    """
    A synaptic cable that adjusts its efficacies in accordance with BCM
    (Bienenstock-Cooper-Munro) theory.

    | --- Synapse Compartments: ---
    | inputs - input (takes in external signals)
    | outputs - output signal (transformation induced by synapses)
    | weights - current value matrix of synaptic efficacies
    | --- Synaptic Plasticity Compartments: ---
    | pre - pre-synaptic spike to drive 1st term of BCM update
    | post - post-synaptic spike to drive 2nd term of BCM update
    | dWeights - current delta matrix containing changes to be applied to synapses

    | References:
    | Bienenstock, E. L., Cooper, L. N, and Munro, P. W. (1982). Theory for the
    | development of neuron selectivity: orientation specificity and binocular
    | interaction in visual cortex. Journal of Neuroscience, 2:32–48.

    Args:
        name: the string name of this cell

        shape: tuple specifying shape of this synaptic cable (usually a 2-tuple
            with number of inputs by number of outputs)

        tau_w: synaptic update time constant

        tau_theta: threshold variable evolution time constant

        w_bound: maximum value to enforce over newly computed efficacies
            (default: 0.); must > 0. to be used

        w_decay: synaptic decay factor (default: 0.)

        wInit: a kernel to drive initialization of this synaptic cable's values;
            typically a tuple with 1st element as a string calling the name of
            initialization to use, e.g., ("uniform", -0.1, 0.1) samples U(-1,1)
            for each dimension/value of this cable's underlying value matrix

        Rscale: a fixed scaling factor to apply to synaptic transform
            (Default: 1.), i.e., yields: out = ((W * Rscale) * in)

        key: PRNG key to control determinism of any underlying random values
            associated with this synaptic cable

        directory: string indicating directory on disk to save synaptic parameter
            values to (i.e., initial threshold values and any persistent adaptive
            threshold values)
    """

    # Define Functions
    def __init__(self, name, shape, tau_w, tau_theta, w_bound=0., w_decay=0.,
                 wInit=("uniform", 0.025, 0.8), Rscale=1., key=None,
                 directory=None, **kwargs):
        super().__init__(name, **kwargs)

        ## constructor-only rng setup
        tmp_key = random.PRNGKey(time.time_ns()) if key is None else key

        ## synapse and BCM hyper-parameters
        self.shape = shape ## shape of synaptic efficacy matrix
        self.tau_w = tau_w ## time constant governing synaptic plasticity
        self.tau_theta = tau_theta ## time constant of threshold delta variables
        self.w_decay = w_decay ## synaptic decay factor
        self.w_bound = w_bound  ## soft weight constraint
        self.Rscale = Rscale ## post-transformation scale factor
        self.theta0 = -1. ## initial condition for theta/threshold variables

        tmp_key, subkey = random.split(tmp_key)
        weights = initialize_params(subkey, wInit, shape)

        self.batch_size = 1
        ## Compartment setup
        preVals = jnp.zeros((self.batch_size, shape[0]))
        postVals = jnp.zeros((self.batch_size, shape[1]))
        self.inputs = Compartment(preVals)
        self.outputs = Compartment(postVals)
        self.pre = Compartment(preVals) ## pre-synaptic statistic
        self.post = Compartment(postVals) ## post-synaptic statistic
        self.theta = Compartment(postVals + self.theta0) ## synaptic threshold variables
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
    def _evolve(t, dt, tau_w, tau_theta, w_bound, w_decay, pre, post, theta, weights):
        weights, theta, dWeights = evolve(dt, pre, post, theta, weights, tau_w,
                                          tau_theta, w_bound, w_decay)
        return weights, theta, dWeights

    @resolver(_evolve)
    def evolve(self, weights, theta, dWeights):
        self.weights.set(weights)
        self.theta.set(theta)
        self.dWeights.set(dWeights)

    @staticmethod
    def _reset(batch_size, shape):
        preVals = jnp.zeros((batch_size, shape[0]))
        postVals = jnp.zeros((batch_size, shape[1]))
        inputs = preVals
        outputs = postVals
        pre = preVals
        post = postVals
        dWeights = jnp.zeros(shape)
        return inputs, outputs, pre, post, dWeights

    @resolver(_reset)
    def reset(self, inputs, outputs, pre, post, dWeights):
        self.inputs.set(inputs)
        self.outputs.set(outputs)
        self.pre.set(pre)
        self.post.set(post)
        self.dWeights.set(dWeights)

    def save(self, directory, **kwargs):
        file_name = directory + "/" + self.name + ".npz"
        jnp.savez(file_name,
                  weights=self.weights.value, theta=self.theta.value)

    def load(self, directory, **kwargs):
        file_name = directory + "/" + self.name + ".npz"
        data = jnp.load(file_name)
        self.weights.set(data['weights'])
        self.theta.set(data['theta'])

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
        Wab = BCMSynapse("Wab", (2, 3), 0.0004, 1, 1)
    print(Wab)
