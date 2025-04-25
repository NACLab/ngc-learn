from jax import random, numpy as jnp, jit
from ngcsimlib.compilers.process import transition
from ngcsimlib.component import Component
from ngcsimlib.compartment import Compartment

from ngclearn.utils.weight_distribution import initialize_params
from ngcsimlib.logger import info
from ngclearn.components.synapses import DenseSynapse
from ngclearn.utils import tensorstats

class ExponentialSynapse(DenseSynapse): ## dynamic exponential synapse cable
    """
    A dynamic exponential synaptic cable; this synapse evolves according to exponential synaptic conductance dynamics.
    Specifically, the dynamics are as follows:

    |  g = g + (weight * gbase) // on the occurrence of a pulse
    |  i = g * (erev - v), where:  d g /dt = -g / tauDecay


    | --- Synapse Compartments: ---
    | inputs - input (takes in external signals)
    | outputs - output signals
    | weights - current value matrix of synaptic efficacies
    | biases - current value vector of synaptic bias values
    | --- Short-Term Plasticity Compartments: ---
    | resources - fixed value matrix of synaptic resources (U)
    | u - release probability; fraction of resources ready for use
    | x - fraction of resources available after neurotransmitter depletion

    | Dynamics note:
    | If tau_d >> tau_f and resources U are large, then synapse is STD-dominated
    | If tau_d << tau_f and resources U are small, then synases is STF-dominated

    Args:
        name: the string name of this cell

        shape: tuple specifying shape of this synaptic cable (usually a 2-tuple
            with number of inputs by number of outputs)

        weight_init: a kernel to drive initialization of this synaptic cable's values;
            typically a tuple with 1st element as a string calling the name of
            initialization to use

        bias_init: a kernel to drive initialization of biases for this synaptic cable
            (Default: None, which turns off/disables biases)

        resist_scale: a fixed (resistance) scaling factor to apply to synaptic
            transform (Default: 1.), i.e., yields: out = ((W * Rscale) * in)

        p_conn: probability of a connection existing (default: 1.); setting
            this to < 1 and > 0. will result in a sparser synaptic structure
            (lower values yield sparse structure)

        tau_f: short-term facilitation (STF) time constant (default: `750` ms); note
            that setting this to `0` ms will disable STF

        tau_d: shoft-term depression time constant (default: `50` ms); note
            that setting this to `0` ms will disable STD

        resources_int: initialization kernel for synaptic resources matrix
    """

    # Define Functions
    def __init__(self, name, shape, weight_init=None, bias_init=None,
                 resist_scale=1., p_conn=1., tau_f=750., tau_d=50.,
                 resources_init=None, **kwargs):
        super().__init__(name, shape, weight_init, bias_init, resist_scale, p_conn, **kwargs)
        ## STP meta-parameters
        self.resources_init = resources_init
        self.tau_f = tau_f
        self.tau_d = tau_d

        ## Set up short-term plasticity / dynamic synapse compartment values
        tmp_key, *subkeys = random.split(self.key.value, 4)
        preVals = jnp.zeros((self.batch_size, shape[0]))
        self.i = Compartment(preVals) ## electrical current output
        self.g = Compartment(preVals) ## conductance variable


    @transition(output_compartments=["outputs", "i", "g"])
    @staticmethod
    def advance_state(
            tau_f, tau_d, Rscale, inputs, weights, biases, i, g
    ):
        s = inputs

        outputs = None #jnp.matmul(inputs, Wdyn * Rscale) + biases
        return outputs

    @transition(output_compartments=["inputs", "outputs", "i", "g"])
    @staticmethod
    def reset(batch_size, shape):
        preVals = jnp.zeros((batch_size, shape[0]))
        postVals = jnp.zeros((batch_size, shape[1]))
        inputs = preVals
        outputs = postVals
        i = preVals
        g = preVals
        return inputs, outputs, i, g

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
        }
        info = {cls.__name__: properties,
                "compartments": compartment_props,
                "dynamics": "outputs = [(W * Rscale) * inputs] + b; "
                            "dg/dt = ",
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
