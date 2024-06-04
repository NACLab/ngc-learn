from jax import random, numpy as jnp, jit
from ngclearn import resolver, Component, Compartment
from ngclearn.utils import tensorstats
from ngclearn.utils.model_utils import initialize_params
import time

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

class StaticSynapse(Component): ## static non-learnable synaptic cable
    """
    A static (dense) synaptic cable; no form of synaptic evolution/adaptation
    is in-built to this component.

    | --- Synapse Compartments: ---
    | inputs - input (takes in external signals)
    | outputs - output
    | weights - current value matrix of synaptic efficacies

    Args:
        name: the string name of this cell

        shape: tuple specifying shape of this synaptic cable (usually a 2-tuple
            with number of inputs by number of outputs)

        wInit: a kernel to drive initialization of this synaptic cable's values;
            typically a tuple with 1st element as a string calling the name of
            initialization to use, e.g., ("uniform", -0.1, 0.1) samples U(-1,1)
            for each dimension/value of this cable's underlying value matrix

        Rscale: a fixed scaling factor to apply to synaptic transform
            (Default: 1.), i.e., yields: out = ((W * Rscale) * in) + b

        key: PRNG key to control determinism of any underlying random values
            associated with this synaptic cable

        directory: string indicating directory on disk to save synaptic parameter
            values to (i.e., initial threshold values and any persistent adaptive
            threshold values)
    """

    # Define Functions
    def __init__(self, name, shape, wInit=("uniform", 0.025, 0.8), Rscale=1.,
                 key=None, directory=None, **kwargs):
        super().__init__(name, **kwargs)

        tmp_key = random.PRNGKey(time.time_ns()) if key is None else key

        ## Synapse meta-parameters
        self.shape = shape ## shape of synaptic efficacy matrix
        self.Rscale = Rscale ## post-transformation scale factor

        tmp_key, subkey = random.split(tmp_key)
        weights = initialize_params(subkey, wInit, shape)

        self.batch_size = 1
        ## Compartment setup
        preVals = jnp.zeros((self.batch_size, shape[0]))
        postVals = jnp.zeros((self.batch_size, shape[1]))
        self.inputs = Compartment(preVals)
        self.outputs = Compartment(postVals)
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
        Wab = StaticSynapse("Wab", (2, 3))
    print(Wab)
