from jax import random, numpy as jnp, jit
from ngclearn.components.jaxComponent import JaxComponent
from ngclearn.utils import tensorstats
from ngclearn.utils.weight_distribution import initialize_params
from ngcsimlib.logger import info

from ngcsimlib.compartment import Compartment
from ngcsimlib.parser import compilable

class DenseSynapse(JaxComponent): ## base dense synaptic cable
    """
    A dense synaptic cable; no form of synaptic evolution/adaptation
    is in-built to this component.

    | --- Synapse Compartments: ---
    | inputs - input (takes in external signals)
    | outputs - output signals
    | weights - current value matrix of synaptic efficacies (strength values)
    | biases - current value vector of synaptic bias values

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
            transform (Default: 1.), i.e., yields: out = ((W * in) * resist_scale) + bias

        p_conn: probability of a connection existing (default: 1.); setting
            this to < 1 and > 0. will result in a sparser synaptic structure
            (lower values yield sparse structure)
    """

    # Define Functions
    def __init__(
            self, name, shape, weight_init=None, bias_init=None, resist_scale=1., 
            p_conn=1., batch_size=1, **kwargs
    ):
        super().__init__(name, **kwargs)

        self.batch_size = Compartment(batch_size, fixed=True)
        self.weight_init = weight_init
        self.bias_init = bias_init

        ## Synapse meta-parameters
        self.shape = Compartment(shape, fixed=True)
        self.resist_scale = Compartment(resist_scale, fixed=True)

        ## Set up synaptic weight values
        tmp_key, *subkeys = random.split(self.key.get(), 4)

        if self.weight_init is None:
            info(self.name, "is using default weight initializer!")
            self.weight_init = {"dist": "uniform", "amin": 0.025, "amax": 0.8}
        weights = initialize_params(subkeys[0], self.weight_init, shape)

        if 0. < p_conn < 1.: ## only non-zero and <1 probs allowed
            p_mask = random.bernoulli(subkeys[1], p=p_conn, shape=shape)
            weights = weights * p_mask ## sparsify matrix

        ## Compartment setup
        preVals = jnp.zeros((self.batch_size.get(), shape[0]))
        postVals = jnp.zeros((self.batch_size.get(), shape[1]))

        self.inputs = Compartment(preVals)
        self.outputs = Compartment(postVals)
        self.weights = Compartment(weights)
        ## Set up (optional) bias values
        if self.bias_init is None:
            info(self.name, "is using default bias value of zero (no bias "
                            "kernel provided)!")
        self.biases = Compartment(initialize_params(subkeys[2], bias_init,
                                                    (1, shape[1]))
                                  if bias_init else 0.0)

    @compilable
    def advance_state(self):
        self.outputs.set((jnp.matmul(self.inputs.get(), self.weights.get()) * self.resist_scale.get()) + self.biases.get())

    @compilable
    def reset(self):
        if not self.inputs.targeted:
            self.inputs.set(jnp.zeros((self.batch_size.get(), self.shape.get()[0])))

        self.outputs.set(jnp.zeros((self.batch_size.get(), self.shape.get()[1])))

    @classmethod
    def help(cls): ## component help function
        properties = {
            "synapse_type": "DenseSynapse - performs a synaptic transformation "
                            "of inputs to produce  output signals (e.g., a "
                            "scaled linear multivariate transformation)"
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
            "batch_size": "Batch size dimension of this component",
            "weight_init": "Initialization conditions for synaptic weight (W) values",
            "bias_init": "Initialization conditions for bias/base-rate (b) values",
            "resist_scale": "Resistance level scaling factor (Rscale); applied to output of transformation",
            "p_conn": "Probability of a connection existing (otherwise, it is masked to zero)"
        }
        info = {cls.__name__: properties,
                "compartments": compartment_props,
                "dynamics": "outputs = [W * inputs] * Rscale + b",
                "hyperparameters": hyperparams}
        return info

if __name__ == '__main__':
    from ngcsimlib.context import Context
    with Context("Bar") as bar:
        Wab = DenseSynapse("Wab", (2, 3))
    print(Wab)
