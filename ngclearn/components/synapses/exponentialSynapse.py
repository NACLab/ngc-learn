from jax import random, numpy as jnp, jit

from ngclearn import compilable #from ngcsimlib.parser import compilable
from ngclearn import Compartment #from ngcsimlib.compartment import Compartment
from ngclearn.components.synapses import DenseSynapse

class ExponentialSynapse(DenseSynapse): ## dynamic exponential synapse cable
    """
    A dynamic exponential synaptic cable; this synapse evolves according to exponential synaptic conductance dynamics.
    Specifically, the conductance dynamics are as follows:

    |  dg/dt = -g/tau_decay + gBar sum_k (t - t_k)
    |  i_syn = g * (syn_rest - v)  // g is `g_syn` in this synapse implementation
    |  where: syn_rest is the post-synaptic reverse potential for this synapse
    |         t_k marks time of -pre-synaptic k-th pulse received by post-synaptic unit


    | --- Synapse Compartments: ---
    | inputs - input (takes in external signals, e.g., pre-synaptic pulses/spikes)
    | outputs - output signals (also equal to i_syn, total electrical current)
    | v - coupled voltages from post-synaptic neurons this synaptic cable connects to
    | weights - current value matrix of synaptic efficacies
    | biases - current value vector of synaptic bias values
    | --- Dynamic / Short-term Plasticity Compartments: ---
    | g_syn - fixed value matrix of synaptic resources (U)
    | i_syn - derived total electrical current variable

    Args:
        name: the string name of this synapse

        shape: tuple specifying shape of this synaptic cable (usually a 2-tuple
            with number of inputs by number of outputs)

        tau_decay: synaptic decay time constant (ms)

        g_syn_bar: maximum conductance elicited by each incoming spike ("synaptic weight")

        syn_rest: synaptic reversal potential; note, if this is set to `None`, then this 
            synaptic conductance model will no longer be voltage-dependent (and will ignore 
            the voltage compartment provided by an external spiking cell)

        weight_init: a kernel to drive initialization of this synaptic cable's values;
            typically a tuple with 1st element as a string calling the name of
            initialization to use

        bias_init: a kernel to drive initialization of biases for this synaptic cable
            (Default: None, which turns off/disables biases) <unused>

        resist_scale: a fixed (resistance) scaling factor to apply to synaptic
            transform (Default: 1.), i.e., yields: out = ((W * Rscale) * in)

        p_conn: probability of a connection existing (default: 1.); setting
            this to < 1 and > 0. will result in a sparser synaptic structure
            (lower values yield sparse structure)

        is_nonplastic: boolean indicating if this synapse permits plasticity adjustments (Default: True)

    """

    def __init__(
            self, name, shape, tau_decay, g_syn_bar, syn_rest, weight_init=None, bias_init=None, resist_scale=1.,
            p_conn=1., is_nonplastic=True, **kwargs
    ):
        super().__init__(name, shape, weight_init, bias_init, resist_scale, p_conn, **kwargs)
        ## dynamic synapse meta-parameters
        self.tau_decay = tau_decay
        self.g_syn_bar = g_syn_bar
        self.syn_rest = syn_rest ## synaptic resting potential

        ## Set up short-term plasticity / dynamic synapse compartment values
        #tmp_key, *subkeys = random.split(self.key.value, 4)
        #preVals = jnp.zeros((self.batch_size, shape[0]))
        postVals = jnp.zeros((self.batch_size, shape[1]))
        self.v = Compartment(postVals)  ## coupled voltage (from a post-synaptic neuron)
        self.i_syn = Compartment(postVals) ## electrical current output
        self.g_syn = Compartment(postVals) ## conductance variable
        if is_nonplastic:
            self.weights.set(self.weights.get() * 0 + 1.)

    @compilable
    def advance_state(self, t, dt):
        s = self.inputs.get()
        ## advance conductance variable
        _out = jnp.matmul(s, self.weights.get()) ## sum all pre-syn spikes at t going into post-neuron)
        dgsyn_dt = -self.g_syn.get()/self.tau_decay + (_out * self.g_syn_bar) * (1./dt)
        g_syn = self.g_syn.get() + dgsyn_dt * dt ## run Euler step to move conductance
        ## compute derive electrical current variable
        i_syn = -g_syn * self.resist_scale
        if self.syn_rest is not None:
            i_syn =  -(g_syn * self.resist_scale) * (self.v.get() - self.syn_rest)
        outputs = i_syn #jnp.matmul(inputs, Wdyn * self.resist_scale) + biases

        self.outputs.set(outputs)
        self.i_syn.set(i_syn)
        self.g_syn.set(g_syn)

    @compilable
    def reset(self):
        preVals = jnp.zeros((self.batch_size.get(), self.shape.get()[0]))
        postVals = jnp.zeros((self.batch_size.get(), self.shape.get()[1]))
        if not self.inputs.targeted:
            self.inputs.set(preVals)
        self.outputs.set(postVals)
        self.i_syn.set(postVals)
        self.g_syn.set(postVals)
        self.v.set(postVals)

    @classmethod
    def help(cls): ## component help function
        properties = {
            "synapse_type": "ExponentialSynapse - performs a synaptic transformation of inputs to produce "
                            "output signals (e.g., a scaled linear multivariate transformation); "
                            "this synapse is dynamic, evolving according to an exponential kernel"
        }
        compartment_props = {
            "inputs":
                {"inputs": "Takes in external input signal values", 
                 "v" : "Post-synaptic voltage dependence (comes from a wired-to spiking cell)"},
            "states":
                {"weights": "Synapse efficacy/strength parameter values",
                 "biases": "Base-rate/bias parameter values",
                 "g_syn" : "Synaptic conductnace",
                 "h_syn" : "Intermediate synaptic conductance",
                 "i_syn" : "Total electrical current", 
                 "key": "JAX PRNG key"},
            "outputs":
                {"outputs": "Output of synaptic transformation"},
        }
        hyperparams = {
            "shape": "Shape of synaptic weight value matrix; number inputs x number outputs",
            "weight_init": "Initialization conditions for synaptic weight (W) values",
            "bias_init": "Initialization conditions for bias/base-rate (b) values",
            "resist_scale": "Resistance level scaling factor (applied to output of transformation)",
            "p_conn": "Probability of a connection existing (otherwise, it is masked to zero)",
            "tau_decay": "Conductance decay time constant (ms)",
            "g_bar_syn": "Maximum conductance value",
            "syn_rest": "Synaptic reversal potential"
        }
        info = {cls.__name__: properties,
                "compartments": compartment_props,
                "dynamics": "outputs = g_syn * (v - syn_rest); "
                            "dgsyn_dt = (W * inputs) * g_syn_bar - g_syn/tau_decay ",
                "hyperparameters": hyperparams}
        return info
