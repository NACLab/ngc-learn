from jax import random, numpy as jnp, jit
from ngclearn import resolver, Component, Compartment
from ngclearn.components.jaxComponent import JaxComponent
from ngclearn.utils import tensorstats
from ngclearn.utils.weight_distribution import initialize_params

@jit
def compute_layer(inp, weight, biases=0., Rscale=1.):
    """
    Applies the transformation/projection induced by the synaptic efficacie
    associated with this synaptic cable

    Args:
        inp: signal input to run through this synaptic cable

        weight: this cable's synaptic value matrix

        biases: this cable's bias value vector (default: 0.)

        Rscale: scale factor to apply to synapses before transform applied
            to input values (default: 1.)

    Returns:
        a projection/transformation of input "inp"
    """
    return jnp.matmul(inp, weight * Rscale) + biases

class DenseSynapse(JaxComponent): ## static non-learnable synaptic cable
    """
    A dense synaptic cable; no form of synaptic evolution/adaptation
    is in-built to this component.

    | --- Synapse Compartments: ---
    | inputs - input (takes in external signals)
    | outputs - output
    | weights - current value matrix of synaptic efficacies
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
            transform (Default: 1.), i.e., yields: out = ((W * Rscale) * in)

        p_conn: probability of a connection existing (default: 1.); setting
            this to < 1 and > 0. will result in a sparser synaptic structure
            (lower values yield sparse structure)
    """

    # Define Functions
    def __init__(self, name, shape, weight_init=None, bias_init=None,
                 resist_scale=1., p_conn=1., **kwargs):
        super().__init__(name, **kwargs)

        self.weight_init = weight_init
        self.bias_init = bias_init

        ## Synapse meta-parameters
        self.shape = shape ## shape of synaptic efficacy matrix
        self.Rscale = resist_scale ## post-transformation scale factor

        ## Set up synaptic weight values
        tmp_key, *subkeys = random.split(self.key.value, 4)
        weights = initialize_params(subkeys[0], weight_init, shape)
        if 0. < p_conn < 1.: ## only non-zero and <1 probs allowed
            mask = random.bernoulli(subkeys[1], p=p_conn, shape=shape)
            weights = weights * mask ## sparsify matrix

        self.batch_size = 1
        ## Compartment setup
        preVals = jnp.zeros((self.batch_size, shape[0]))
        postVals = jnp.zeros((self.batch_size, shape[1]))
        self.inputs = Compartment(preVals)
        self.outputs = Compartment(postVals)
        self.weights = Compartment(weights)
        ## Set up (optional) bias values
        self.biases = Compartment(initialize_params(subkeys[2], bias_init,
                                                    (1, shape[1]))
                                  if bias_init else 0.0)

    @staticmethod
    def _advance_state(t, dt, Rscale, inputs, weights, biases):
        outputs = compute_layer(inputs, weights, biases, Rscale)
        return outputs

    @resolver(_advance_state)
    def advance_state(self, outputs):
        self.outputs.set(outputs)

    @staticmethod
    def _reset(batch_size, shape):
        preVals = jnp.zeros((batch_size, shape[0]))
        postVals = jnp.zeros((batch_size, shape[1]))
        inputs = preVals
        outputs = postVals
        return inputs, outputs

    @resolver(_reset)
    def reset(self, inputs, outputs):
        self.inputs.set(inputs)
        self.outputs.set(outputs)

    def save(self, directory, **kwargs):
        file_name = directory + "/" + self.name + ".npz"
        if self.bias_init != None:
            jnp.savez(file_name, weights=self.weights.value,
                      biases=self.biases.value)
        else:
            jnp.savez(file_name, weights=self.weights.value)

    def load(self, directory, **kwargs):
        file_name = directory + "/" + self.name + ".npz"
        data = jnp.load(file_name)
        self.weights.set(data['weights'])
        if "biases" in data.keys():
            self.biases.set(data['biases'])

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
        Wab = DenseSynapse("Wab", (2, 3))
    print(Wab)
