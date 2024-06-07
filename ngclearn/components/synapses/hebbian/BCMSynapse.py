from jax import random, numpy as jnp, jit
from ngclearn import resolver, Component, Compartment
from ngclearn.components.synapses import DenseSynapse
from ngclearn.utils import tensorstats

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

class BCMSynapse(DenseSynapse): # BCM-adjusted synaptic cable
    """
    A synaptic cable that adjusts its efficacies in accordance with BCM
    (Bienenstock-Cooper-Munro) theory.

    | --- Synapse Compartments: ---
    | inputs - input (takes in external signals)
    | outputs - output signal (transformation induced by synapses)
    | weights - current value matrix of synaptic efficacies
    | key - JAX RNG key
    | --- Synaptic Plasticity Compartments: ---
    | pre - pre-synaptic spike to drive 1st term of BCM update (takes in external signals)
    | post - post-synaptic spike to drive 2nd term of BCM update (takes in external signals)
    | theta - synaptic threshold (post-synaptic) variables
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

        weight_init: a kernel to drive initialization of this synaptic cable's values;
            typically a tuple with 1st element as a string calling the name of
            initialization to use

        resist_scale: a fixed scaling factor to apply to synaptic transform
            (Default: 1.), i.e., yields: out = ((W * Rscale) * in)

        p_conn: probability of a connection existing (default: 1.); setting
            this to < 1. will result in a sparser synaptic structure
    """

    # Define Functions
    def __init__(self, name, shape, tau_w, tau_theta, w_bound=0., w_decay=0.,
                 weight_init=None, resist_scale=1., p_conn=1., **kwargs):
        super().__init__(name, shape, weight_init, None, resist_scale, p_conn, **kwargs)

        ## Synapse and BCM hyper-parameters
        self.shape = shape ## shape of synaptic efficacy matrix
        self.tau_w = tau_w ## time constant governing synaptic plasticity
        self.tau_theta = tau_theta ## time constant of threshold delta variables
        self.w_decay = w_decay ## synaptic decay factor
        self.w_bound = w_bound  ## soft weight constraint
        self.Rscale = resist_scale ## post-transformation scale factor
        self.theta0 = -1. ## initial condition for theta/threshold variables

        self.batch_size = 1
        ## Compartment setup
        preVals = jnp.zeros((self.batch_size, shape[0]))
        postVals = jnp.zeros((self.batch_size, shape[1]))
        self.pre = Compartment(preVals) ## pre-synaptic statistic
        self.post = Compartment(postVals) ## post-synaptic statistic
        self.theta = Compartment(postVals + self.theta0) ## synaptic threshold variables
        self.dWeights = Compartment(self.weights.value * 0)

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
