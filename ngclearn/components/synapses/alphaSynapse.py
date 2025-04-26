from jax import random, numpy as jnp, jit
from ngcsimlib.compilers.process import transition
from ngcsimlib.component import Component
from ngcsimlib.compartment import Compartment

from ngclearn.utils.weight_distribution import initialize_params
from ngcsimlib.logger import info
from ngclearn.components.synapses import DenseSynapse
from ngclearn.utils import tensorstats

class AlphaSynapse(DenseSynapse): ## dynamic alpha synapse cable
    """
    A dynamic alpha synaptic cable; this synapse evolves according to alpha synaptic conductance dynamics.
    Specifically, the conductance dynamics are as follows:

    |  dh/dt = -h/tau_syn + gBar sum_k (t - t_k) // h is an intermediate variable
    |  dg/dt = -g/tau_syn + h/tau_syn 
    |  i_syn = g * (syn_rest - v)  // g is `g_syn` and h is `h_syn` in this synapse implementation
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

        tau_syn: synaptic time constant (ms)

        g_syn_bar: maximum conductance elicited by each incoming spike ("synaptic weight")

        syn_rest: synaptic reversal potential

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

    # Define Functions
    def __init__(
            self, name, shape, tau_syn, g_syn_bar, syn_rest, weight_init=None, bias_init=None, resist_scale=1., p_conn=1.,
            is_nonplastic=True, **kwargs
    ):
        super().__init__(name, shape, weight_init, bias_init, resist_scale, p_conn, **kwargs)
        ## dynamic synapse meta-parameters
        self.tau_syn = tau_syn
        self.g_syn_bar = g_syn_bar
        self.syn_rest = syn_rest ## synaptic resting potential

        ## Set up short-term plasticity / dynamic synapse compartment values
        #tmp_key, *subkeys = random.split(self.key.value, 4)
        #preVals = jnp.zeros((self.batch_size, shape[0]))
        postVals = jnp.zeros((self.batch_size, shape[1]))
        self.v = Compartment(postVals)  ## coupled voltage (from a post-synaptic neuron)
        self.i_syn = Compartment(postVals) ## electrical current output
        self.g_syn = Compartment(postVals) ## conductance variable
        self.h_syn = Compartment(postVals) ## intermediate conductance variable
        if is_nonplastic:
            self.weights.set(self.weights.value * 0 + 1.)

    @transition(output_compartments=["outputs", "i_syn", "g_syn", "h_syn"])
    @staticmethod
    def advance_state(
            dt, tau_syn, g_syn_bar, syn_rest, Rscale, inputs, weights, i_syn, g_syn, h_syn, v
    ):
        s = inputs
        ## advance conductance variable
        _out = jnp.matmul(s, weights) ## sum all pre-syn spikes at t going into post-neuron)
        dhsyn_dt = -h_syn/tau_syn + _out * g_syn_bar
        h_syn = h_syn + dhsyn_dt * dt ## run Euler step to move intermediate conductance h

        dgsyn_dt = -g_syn/tau_syn + h_syn # or -g_syn/tau_syn + h_syn/tau_syn
        g_syn = g_syn + dgsyn_dt * dt ## run Euler step to move conductance g

        i_syn =  -g_syn * (v - syn_rest)
        outputs = i_syn #jnp.matmul(inputs, Wdyn * Rscale) + biases
        return outputs, i_syn, g_syn, h_syn

    @transition(output_compartments=["inputs", "outputs", "i_syn", "g_syn", "h_syn", "v"])
    @staticmethod
    def reset(batch_size, shape):
        preVals = jnp.zeros((batch_size, shape[0]))
        postVals = jnp.zeros((batch_size, shape[1]))
        inputs = preVals
        outputs = postVals
        i_syn = postVals
        g_syn = postVals
        h_syn = postVals
        v = postVals
        return inputs, outputs, i_syn, g_syn, h_syn, v

    def save(self, directory, **kwargs):
        file_name = directory + "/" + self.name + ".npz"
        if self.bias_init != None:
            jnp.savez(file_name, weights=self.weights.value, biases=self.biases.value)
        else:
            jnp.savez(file_name, weights=self.weights.value)

    def load(self, directory, **kwargs):
        file_name = directory + "/" + self.name + ".npz"
        data = jnp.load(file_name)
        self.weights.set(data['weights'])
        if "biases" in data.keys():
            self.biases.set(data['biases'])

    @classmethod
    def help(cls): ## component help function
        properties = {
            "synapse_type": "STPDenseSynapse - performs a synaptic transformation of inputs to produce "
                            "output signals (e.g., a scaled linear multivariate transformation); "
                            "this synapse is dynamic, adapting via a form of short-term plasticity"
        }
        compartment_props = {
            "inputs":
                {"inputs": "Takes in external input signal values"},
            "states":
                {"weights": "Synapse efficacy/strength parameter values",
                 "biases": "Base-rate/bias parameter values",
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
            "tau_syn": "Synaptic time constant (ms)",
            "g_bar_syn": "Maximum conductance value",
            "syn_rest": "Synaptic reversal potential"
        }
        info = {cls.__name__: properties,
                "compartments": compartment_props,
                "dynamics": "outputs = g_syn * (v - syn_rest); "
                            "dgsyn_dt = (W * inputs) * g_syn_bar - g_syn/tau_syn ",
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
