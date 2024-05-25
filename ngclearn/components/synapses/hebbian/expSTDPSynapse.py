# %%

from ngcsimlib.component import Component
from ngcsimlib.compartment import Compartment
from ngcsimlib.resolver import resolver
from jax import random, numpy as jnp, jit
from functools import partial
from ngclearn.utils import tensorstats
from ngclearn.utils.model_utils import initialize_params, normalize_matrix
import time

@partial(jit, static_argnums=[6,7,8,9,10,11,12])
def evolve(dt, pre, x_pre, post, x_post, W, w_bound=1., eta=0.00005,
            x_tar=0.7, exp_beta=1., Aplus=1., Aminus=0., w_norm=None):
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

        exp_beta: controls effect of exponential Hebbian shift/dependency

        Aplus: strength of long-term potentiation (LTP)

        Aminus: strength of long-term depression (LTD)

        w_norm: if not None, applies an L2 norm constraint to synapses

    Returns:
        the newly evolved synaptic weight value matrix, synaptic update matrix
    """
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
    return _W, dW

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

class ExpSTDPSynapse(Component):
    """
    A synaptic cable that adjusts its efficacies via trace-based form of
    spike-timing-dependent plasticity (STDP) based on an exponential weight
    dependence (the strength of which is controlled by a factor).

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

        eta: global learning rate

        exp_beta: controls effect of exponential Hebbian shift/dependency

        Aplus: strength of long-term potentiation (LTP)

        Aminus: strength of long-term depression (LTD)

        preTrace_target: controls degree of pre-synaptic disconnect, i.e., amount of decay
                 (higher -> lower synaptic values)

        wInit: a kernel to drive initialization of this synaptic cable's values;
            typically a tuple with 1st element as a string calling the name of
            initialization to use, e.g., ("uniform", -0.1, 0.1) samples U(-1,1)
            for each dimension/value of this cable's underlying value matrix

        Rscale: a fixed scaling (resistance) factor to apply to synaptic transform
            (Default: 1.), i.e., yields: out = ((W * Rscale) * in) + b

        key: PRNG key to control determinism of any underlying random values
            associated with this synaptic cable

        useVerboseDict: triggers slower, verbose dictionary mode (Default: False)

        directory: string indicating directory on disk to save synaptic parameter
            values to (i.e., initial threshold values and any persistent adaptive
            threshold values)
    """

    # Define Functions
    def __init__(self, name, shape, eta, exp_beta, Aplus, Aminus,
                 preTrace_target, wInit=("uniform", 0.025, 0.8), Rscale=1.,
                 key=None, directory=None, **kwargs):
        super().__init__(name, **kwargs)

        tmp_key = random.PRNGKey(time.time_ns()) if key is None else key

        ## Exp-STDP meta-parameters
        self.shape = shape ## shape of synaptic efficacy matrix
        self.eta = eta ## global learning rate governing plasticity
        self.exp_beta = exp_beta ## if not None, will trigger exp-depend STPD rule
        self.preTrace_target = preTrace_target ## target (pre-synaptic) trace activity value # 0.7
        self.Aplus = Aplus ## LTP strength
        self.Aminus = Aminus ## LTD strength
        self.shape = shape  ## shape of synaptic matrix W
        self.Rscale = Rscale ## post-transformation scale factor
        self.w_bound = 1. ## soft weight constraint

        tmp_key, subkey = random.split(tmp_key)
        #self.weights = random.uniform(subkey, shape, minval=lb, maxval=ub)
        weights = initialize_params(subkey, wInit, shape)

        self.batch_size = 1
        ## Compartment setup
        preVals = jnp.zeros((self.batch_size, shape[0]))
        postVals = jnp.zeros((self.batch_size, shape[1]))
        self.inputs = Compartment(preVals)
        self.outputs = Compartment(postVals)
        self.preSpike = Compartment(preVals)
        self.postSpike = Compartment(postVals)
        self.preTrace = Compartment(preVals)
        self.postTrace = Compartment(postVals)
        self.weights = Compartment(weights)

    @staticmethod
    def _advance_state(t, dt, Rscale, inputs, weights):
        ## run signals across synapses
        outputs = compute_layer(inputs, weights, Rscale)
        return outputs

    @resolver(_advance_state)
    def advance_state(self, outputs):
        self.outputs.set(outputs)

    @staticmethod
    def _evolve(t, dt, w_bound, eta, preTrace_target, exp_beta, Aplus, Aminus,
                    preSpike, postSpike, preTrace, postTrace, weights):
        weights, dW = evolve(dt, preSpike, preTrace, postSpike, postTrace, weights,
                             w_bound=w_bound, eta=eta, x_tar=preTrace_target,
                             exp_beta=exp_beta, Aplus=Aplus, Aminus=Aminus)
        return weights

    @resolver(_evolve)
    def evolve(self, weights):
        self.weights.set(weights)

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
        return inputs, outputs, preSpike, postSpike, preTrace, postTrace

    @resolver(_reset)
    def reset(self, inputs, outputs, preSpike, postSpike, preTrace, postTrace):
        inputs.set(inputs)
        outputs.set(outputs)
        preSpike.set(preSpike)
        postSpike.set(postSpike)
        preTrace.set(preTrace)
        postTrace.set(postTrace)

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
        Wab = ExpSTDPSynapse("Wab", (2, 3), 0.0004, 1, 1, 1, 1)
    print(Wab)