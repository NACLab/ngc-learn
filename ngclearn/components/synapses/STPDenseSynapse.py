from jax import random, numpy as jnp, jit
from ngcsimlib.logger import info

from ngclearn.utils.distribution_generator import DistributionGenerator
from ngclearn import compilable #from ngcsimlib.parser import compilable
from ngclearn import Compartment #from ngcsimlib.compartment import Compartment
from ngclearn.components.synapses import DenseSynapse

class STPDenseSynapse(DenseSynapse): ## short-term plastic synaptic cable
    """
    A dynamic dense synaptic cable; this synapse evolves according to
    short-term plasticity (STP) dynamics.

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

    def __init__(
            self, name, shape, weight_init=None, bias_init=None, resist_scale=1., p_conn=1., tau_f=750., tau_d=50.,
            resources_init=None, **kwargs
    ):
        super().__init__(name, shape, weight_init, bias_init, resist_scale, p_conn, **kwargs)
        ## STP meta-parameters
        self.resources_init = resources_init
        self.tau_f = tau_f
        self.tau_d = tau_d

        ## Set up short-term plasticity / dynamic synapse compartment values
        tmp_key, *subkeys = random.split(self.key.get(), 4)
        preVals = jnp.zeros((self.batch_size, shape[0]))
        self.u = Compartment(preVals) ## release prob variables
        self.x = Compartment(preVals + 1) ## resource availability variables
        self.Wdyn = Compartment(self.weights.get() * 0) ## dynamic synapse values
        if self.resources_init is None:
            info(self.name, "is using default resources value initializer!")
            #self.resources_init = {"dist": "uniform", "amin": 0.125, "amax": 0.175} # 0.15
            self.resources_init = DistributionGenerator.uniform(low=0.125, high=0.175)
        self.resources = Compartment(
            self.resources_init(shape, subkeys[2]) #initialize_params(subkeys[2], self.resources_init, shape)
        ) ## matrix U - synaptic resources matrix

    @compilable
    def advance_state(self, t, dt):
        s = self.inputs.get()
        ## compute short-term facilitation
        #u = u - u * (1./tau_f) + (resources * (1. - u)) * s
        if self.tau_f > 0.: ## compute short-term facilitation
            u = self.u.get() - self.u.get() * (1./self.tau_f) + (self.resources.get() * (1. - self.u.get())) * s
        else:
            u = self.resources.get() ## disabling STF yields fixed resource u variables
        ## compute dynamic synaptic values/conductances
        Wdyn = (self.weights.get() * u * self.x.get()) * s + self.Wdyn.get() * (1. - s) ## OR: -W/tau_w + W * u * x
        ## compute short-term depression
        x = self.x.get()
        if self.tau_d > 0.:
            x = x + (1. - x) * (1./self.tau_d) - u * x * s
        ## else, do nothing with x (keep it pointing to current x compartment)
        outputs = jnp.matmul(self.inputs.get(), Wdyn * self.resist_scale) + self.biases.get()

        self.outputs.set(outputs)
        self.u.set(u)
        self.x.set(x)
        self.Wdyn.set(Wdyn)

    @compilable
    def reset(self):
        preVals = jnp.zeros((self.batch_size.get(), self.shape.get()[0]))
        postVals = jnp.zeros((self.batch_size.get(), self.shape.get()[1]))
        if not self.inputs.targeted:
            self.inputs.set(preVals)
        self.outputs.set(postVals)
        self.u.set(preVals)
        self.x.set(preVals + 1)
        self.Wdyn.set(jnp.zeros(self.shape.get()))

    # def save(self, directory, **kwargs):
    #     file_name = directory + "/" + self.name + ".npz"
    #     if self.bias_init != None:
    #         jnp.savez(file_name,
    #                   weights=self.weights.value,
    #                   biases=self.biases.value,
    #                   resources=self.resources.value)
    #     else:
    #         jnp.savez(file_name,
    #                   weights=self.weights.value,
    #                   resources=self.resources.value)
    #
    # def load(self, directory, **kwargs):
    #     file_name = directory + "/" + self.name + ".npz"
    #     data = jnp.load(file_name)
    #     self.weights.set(data['weights'])
    #     self.resources.set(data['resources'])
    #     if "biases" in data.keys():
    #         self.biases.set(data['biases'])

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
                 "resources": "Synaptic resource paramter values (U)",
                 "u": "Release probability variables",
                 "x": "Resource depletion variables",
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
            "tau_f": "Short-term facilitation time constant",
            "tau_d": "Short-term depression time constant"
        }
        info = {cls.__name__: properties,
                "compartments": compartment_props,
                "dynamics": "outputs = [(W * Rscale) * inputs] + b; "
                            "dW/dt = W_full * u * x * inputs",
                "hyperparameters": hyperparams}
        return info
