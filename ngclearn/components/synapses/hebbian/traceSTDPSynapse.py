from ngcsimlib.component import Component
from ngcsimlib.compartment import Compartment
from ngcsimlib.resolver import resolver
from jax import random, numpy as jnp, jit
from functools import partial
from ngclearn.utils.model_utils import initialize_params
import time

#@partial(jit, static_argnums=[6,7,8,9,10,11])
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
    dW = (dWpost + dWpre) * eta
    _W = W + dW # do a gradient ascent update/shift
    eps = 0.01 # 0.001
    _W = jnp.clip(_W, eps, 1. - eps) #jnp.abs(w_bound)) # 0.01, w_bound) ## enforce non-negativity
    #print(_W)
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

class TraceSTDPSynapse(Component): # power-law / trace-based STDP
    """
    A synaptic cable that adjusts its efficacies via trace-based form of
    spike-timing-dependent plasticity (STDP), including an optional power-scale
    dependence that can be equipped to the Hebbian adjustment (the strength of
    which is controlled by a scalar factor).

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

        eta: global learning rate

        Aplus: strength of long-term potentiation (LTP)

        Aminus: strength of long-term depression (LTD)

        mu: controls the power scale of the Hebbian shift

        preTrace_target: controls degree of pre-synaptic disconnect, i.e., amount of decay
                 (higher -> lower synaptic values)

        wInit: a kernel to drive initialization of this synaptic cable's values;
            typically a tuple with 1st element as a string calling the name of
            initialization to use, e.g., ("uniform", -0.1, 0.1) samples U(-1,1)
            for each dimension/value of this cable's underlying value matrix

        Rscale: a fixed scaling factor to apply to synaptic transform
            (Default: 1.), i.e., yields: out = ((W * Rscale) * in) + b

        key: PRNG key to control determinism of any underlying random values
            associated with this synaptic cable

        useVerboseDict: triggers slower, verbose dictionary mode (Default: False)

        directory: string indicating directory on disk to save synaptic parameter
            values to (i.e., initial threshold values and any persistent adaptive
            threshold values)
    """

    # Define Functions
    def __init__(self, name, shape, eta, Aplus, Aminus, mu=0.,
                 preTrace_target=0., wInit=("uniform", 0.025, 0.8), Rscale=1., 
                 key=None, useVerboseDict=False, directory=None, **kwargs):
        super().__init__(name, useVerboseDict, **kwargs)

        ## constructor-only rng setup
        tmp_key = random.PRNGKey(time.time_ns()) if key is None else key

        ##parms
        self.shape = shape ## shape of synaptic efficacy matrix
        self.eta = eta ## global learning rate governing plasticity
        self.mu = mu ## controls power-scaling of STDP rule
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
    def _evolve(t, dt, w_bound, eta, preTrace_target, mu, Aplus, Aminus,
                preSpike, postSpike, preTrace, postTrace, weights):
        weights, dW = evolve(dt, preSpike, preTrace, postSpike, postTrace, weights,
                             w_bound=w_bound, eta=eta, x_tar=preTrace_target, mu=mu,
                             Aplus=Aplus, Aminus=Aminus)
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
        jnp.savez(file_name, 
                  weights=self.weights.value)

    def load(self, directory, **kwargs):
        file_name = directory + "/" + self.name + ".npz"
        data = jnp.load(file_name)
        self.weights.set( data['weights'] )

# Testing
if __name__ == '__main__':
    from ngcsimlib.compartment import All_compartments
    from ngcsimlib.context import Context
    from ngcsimlib.commands import Command

    def wrapper(compiled_fn):
        def _wrapped(*args):
            # vals = jax.jit(compiled_fn)(*args, compartment_values={key: c.value for key, c in All_compartments.items()})
            vals = compiled_fn(*args, compartment_values={key: c.value for key, c in All_compartments.items()})
            for key, value in vals.items():
                All_compartments[str(key)].set(value)
            return vals
        return _wrapped

    class AdvanceCommand(Command):
        compile_key = "advance"
        def __call__(self, t=None, dt=None, *args, **kwargs):
            for component in self.components:
                component.gather()
                component.advance(t=t, dt=dt)

    class EvolveCommand(Command):
        compile_key = "evolve"
        def __call__(self, t=None, dt=None, *args, **kwargs):
            for component in self.components:
                component.evolve(t=t, dt=dt)

    class ResetCommand(Command):
        compile_key = "reset"
        def __call__(self, t=None, dt=None, *args, **kwargs):
            for component in self.components:
                component.reset(t=t, dt=dt)

    dkey = random.PRNGKey(1234)
    with Context("Context") as context:
        W = TraceSTDPSynapse("W", shape=(1,1), eta=0.1, Aplus=1., Aminus=0.9, mu=0.,
                             preTrace_target=0.0, wInit=("uniform", 0.025, 0.8),
                             key=dkey) #78.5, norm_T=250)
        advance_cmd = AdvanceCommand(components=[W], command_name="Advance")
        evolve_cmd = EvolveCommand(components=[W], command_name="Evolve")
        reset_cmd = ResetCommand(components=[W], command_name="Reset")

    T = 30 #250
    dt = 1.

    compiled_advance_cmd, _ = advance_cmd.compile()
    wrapped_advance_cmd = wrapper(jit(compiled_advance_cmd))

    compiled_evolve_cmd, _ = evolve_cmd.compile()
    wrapped_evolve_cmd = wrapper(jit(compiled_evolve_cmd))

    compiled_reset_cmd, _ = reset_cmd.compile()
    wrapped_reset_cmd = wrapper(jit(compiled_reset_cmd))

    t = 0.
    for i in range(T): # i is "t"
        val = ((i % 2 == 0)) * 1.
        pre_spk = jnp.asarray([[val]])
        post_spk = pre_spk
        pre_tr = post_tr = pre_spk
        W.inputs.set(pre_spk)
        W.preSpike.set(pre_spk)
        W.preTrace.set(pre_tr)
        W.postSpike.set(post_spk)
        W.postTrace.set(post_tr)
        wrapped_advance_cmd(t, dt) ## pass in t and dt and run step forward of simulation
        wrapped_evolve_cmd(t, dt) ## pass in t and dt and run step forward of simulation
        t = t + dt
        print(f"---[ Step {i} ]---")
        print(f"[W] in: {W.inputs.value}, out: {W.outputs.value}, preS: {W.preSpike.value}, " \
              f"preTr: {W.preTrace.value}, postS: {W.postSpike.value}, postTr: {W.postTrace.value}," \
              f"W: {W.weights.value}")
    #a.reset()
    wrapped_reset_cmd()
    print(f"---[ After reset ]---")
    print(f"[W] in: {W.inputs.value}, out: {W.outputs.value}, preS: {W.preSpike.value}, " \
          f"preTr: {W.preTrace.value}, postS: {W.postSpike.value}, postTr: {W.postTrace.value}," \
          f"W: {W.weights.value}")
