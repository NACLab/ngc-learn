import math

from jax import random, numpy as jnp, jit
from ngclearn import resolver, Component, Compartment
from ngclearn.components.jaxComponent import JaxComponent
from ngclearn.utils import tensorstats
from ngclearn.utils.weight_distribution import initialize_params
from ngcsimlib.logger import info

# patch_shape
# recField_size

# patch == kernel
# RF_shape = patch_shape
# patch_size == Reciptive fields
# stride == overlap
# n_patches = input//stride


def create_multi_patch_synapses(weight_init, num_models, patch_shape, key):
    weight_shape = (num_models * patch_shape[0], num_models * patch_shape[1])
    weights = initialize_params(key[2], {"dist": "constant", "value": 0.}, weight_shape, use_numpy=True)
    for i in range(num_models):
        weights[patch_shape[0] * i:
                patch_shape[0] * (i + 1),
                patch_shape[1] * i:
                patch_shape[1] * (i + 1)] = initialize_params(key[1],
                                                             init_kernel=weight_init,
                                                             shape=patch_shape,
                                                             use_numpy=True)

class DepatchedSynapse(JaxComponent): ## base depatched synaptic cable
    # Define Functions
    def __init__(self, name, shape, model_shape=(1,1), weight_init=None, bias_init=None,
                 resist_scale=1., p_conn=1., batch_size=1, **kwargs):
        super().__init__(name, **kwargs)

        self.batch_size = batch_size
        self.weight_init = weight_init
        self.bias_init = bias_init
        self.model_shape = model_shape

        self.n_models = model_shape[0]                   # = ni = #of parents = n3
        self.model_patches = model_shape[1]              # = ci = #of children per parent = c3

        ## Synapse meta-parameters
        # print(shape, model_shape, self.n_models, self.model_patches, '------', (shape[0], shape[1]*self.model_patches))
        self.sub_shape = (shape[0], shape[1]*self.model_patches)  ## shape of synaptic efficacy matrix           # = (in_dim, hid_dim) = (d3, d2)
        self.shape = (self.sub_shape[0] * self.n_models, self.sub_shape[1] * self.n_models)

        self.sub_shape = self.sub_shape[1], self.sub_shape[0]
        self.shape = self.shape[1], self.shape[0]

        self.Rscale = resist_scale ## post-transformation scale factor
        ## Set up synaptic weight values
        tmp_key, *subkeys = random.split(self.key.value, 4)
        if self.weight_init is None:
            info(self.name, "is using default weight initializer!")
            self.weight_init = {"dist": "uniform", "amin": 0.025, "amax": 0.8}
        else:
            weights = initialize_params(subkeys[0], self.weight_init, self.shape)

        weights = create_multi_patch_synapses(self.weight_init, self.n_models, self.sub_shape, subkeys[1])

        if 0. < p_conn < 1.: ## only non-zero and <1 probs allowed
            mask = random.bernoulli(subkeys[1], p=p_conn, shape=self.shape)
            weights = weights * mask ## sparsify matrix


        self.batch_size = 1
        ## Compartment setup
        preVals = jnp.zeros((self.batch_size, self.shape[0]))
        postVals = jnp.zeros((self.batch_size, self.shape[1]))
        self.inputs = Compartment(preVals)
        self.outputs = Compartment(postVals)
        self.weights = Compartment(weights)
        ## Set up (optional) bias values
        if self.bias_init is None:
            info(self.name, "is using default bias value of zero (no bias "
                            "kernel provided)!")
        self.biases = Compartment(initialize_params(subkeys[2], bias_init,
                                                    (1, self.shape[1]))
                                  if bias_init else 0.0)

    @staticmethod
    def _advance_state(Rscale, inputs, weights, biases):
        outputs = (jnp.matmul(inputs, weights) * Rscale) + biases
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
        Wab = DepatchedSynapse("Wab", (2, 3))
    print(Wab)
