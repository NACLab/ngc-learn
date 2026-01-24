# %%

import matplotlib.pyplot as plt
from jax import random, numpy as jnp, jit
from ngclearn.components.jaxComponent import JaxComponent
from ngclearn.utils.distribution_generator import DistributionGenerator
from ngcsimlib.logger import info
from ngclearn import compilable #from ngcsimlib.parser import compilable
from ngclearn import Compartment #from ngcsimlib.compartment import Compartment

def _create_multi_patch_synapses(key, shape, n_modules, module_stride=(0, 0), initialization_type=DistributionGenerator.fan_in_gaussian()):
    key, *subkey = random.split(key, n_modules+10)

    module_shape = (shape[0] // n_modules, shape[1] // n_modules)
    di, dj = module_shape
    si, sj = module_stride

    module_shape = di + (2 * si), dj + (2 * sj)


    weight_shape = ((n_modules * di) + 2 * si, (n_modules * dj) + 2 * sj)
    weights = jnp.zeros(weight_shape)
    w_masks = jnp.zeros(weight_shape)

    for i in range(n_modules):
        start_i = i * di
        end_i = (i + 1) * di + 2 * si
        start_j = i * dj
        end_j = (i + 1) * dj + 2 * sj

        shape_ = (end_i - start_i, end_j - start_j)  # (di + 2 * si, dj + 2 * sj)

        weights = weights.at[start_i : end_i,
                start_j : end_j].set(initialization_type(shape_, subkey[i]))

        w_masks = w_masks.at[start_i : end_i,
                start_j : end_j].set(jnp.ones(shape_))

    if si!=0:
        weights = weights.at[:si,:].set(0.)
        weights = weights.at[-si:,:].set(0.)

        w_masks = w_masks.at[:si,:].set(0.)
        w_masks = w_masks.at[-si:,:].set(0.)

    if sj!=0:
        weights = weights.at[:,:sj].set(0.)
        weights = weights.at[:, -sj:].set(0.)

        w_masks = weights.at[:,:sj].set(0.)
        w_masks = weights.at[:, -sj:].set(0.)


    return weights, module_shape, w_masks

class PatchedSynapse(JaxComponent): ## base patched synaptic cable
    """
    A patched dense synaptic cables that creates multiple small dense synaptic cables; no form of synaptic evolution/adaptation
    is in-built to this component.

    | --- Synapse Compartments: ---
    | inputs - input (takes in external signals)
    | outputs - output signals (transformation induced by synapses)
    | weights - current value matrix of synaptic efficacies
    | biases - current value vector of synaptic bias values
    | key - JAX PRNG key
    | --- Synaptic Plasticity Compartments: ---
    | pre - pre-synaptic signal to drive first term of Hebbian update (takes in external signals)
    | post - post-synaptic signal to drive 2nd term of Hebbian update (takes in external signals)
    | dWweights - current delta matrix containing changes to be applied to synaptic efficacies
    | dBiases - current delta vector containing changes to be applied to bias values
    | opt_params - locally-embedded optimizer statisticis (e.g., Adam 1st/2nd moments if adam is used)

    Args:
        name: the string name of this cell

        shape: tuple specifying shape of this synaptic cable (usually a 2-tuple
            with number of inputs by number of outputs)

        n_sub_models: The number of submodels in each layer (Default: 1 similar functionality as DenseSynapse)

        stride_shape: Stride shape of overlapping synaptic weight value matrix
            (Default: (0, 0))

        eta: global learning rate

        weight_init: a kernel to drive initialization of this synaptic cable's values;
            typically a tuple with 1st element as a string calling the name of
            initialization to use

        bias_init: a kernel to drive initialization of biases for this synaptic cable
            (Default: None, which turns off/disables biases)

        w_masks: weight mask matrix

        pre_wght: pre-synaptic weighting factor (Default: 1.)

        post_wght: post-synaptic weighting factor (Default: 1.)

        resist_scale: a fixed scaling factor to apply to synaptic transform
            (Default: 1.), i.e., yields: out = ((W * Rscale) * in) + b

        p_conn: probability of a connection existing (default: 1.); setting
            this to < 1. will result in a sparser synaptic structure
    """

    def __init__(self, name, shape, n_sub_models=1, stride_shape=(0,0), weight_init=None, bias_init=None,
            resist_scale=1., p_conn=1., batch_size=1, **kwargs
    ):
        super().__init__(name, **kwargs)

        self.Rscale = resist_scale
        self.batch_size = batch_size
        self.weight_init = weight_init
        self.bias_init = bias_init

        self.n_sub_models = n_sub_models
        self.sub_stride = stride_shape

        tmp_key, *subkeys = random.split(self.key.get(), 4)
        if self.weight_init is None:
            info(self.name, "is using default weight initializer!")
            self.weight_init = DistributionGenerator.fan_in_gaussian()

        weights, self.sub_shape, self.w_masks = _create_multi_patch_synapses(
            key=tmp_key, shape=shape, n_modules=self.n_sub_models, module_stride=self.sub_stride,
            initialization_type = self.weight_init
                                                                             )

        self.shape = weights.shape
        
        if 0. < p_conn < 1.: ## only non-zero and <1 probs allowed
            mask = random.bernoulli(subkeys[1], p=p_conn, shape=self.shape)
            weights = weights * mask ## sparsify matrix

        ## Compartment setup
        preVals = jnp.zeros((self.batch_size, self.shape[0]))
        postVals = jnp.zeros((self.batch_size, self.shape[1]))

        self.inputs = Compartment(preVals)
        self.outputs = Compartment(postVals)
        self.weights = Compartment(weights)

        self.post_in = Compartment(postVals)
        self.pre_out = Compartment(preVals)
        self.weights_T = Compartment(weights.T)

        ## Set up (optional) bias values
        if self.bias_init is None:
            info(self.name, "is using default bias value of zero (no bias "
                            "kernel provided)!")
        self.biases = Compartment(self.bias_init((1, self.shape[1]), subkeys[2]) if bias_init else 0.0)

    @compilable
    def advance_state(self):
        # Get the variables
        inputs = self.inputs.get()
        post_in = self.post_in.get()
        weights = self.weights.get()
        biases = self.biases.get()

        outputs = (jnp.matmul(inputs, weights) * self.Rscale) + biases
        pre_out = jnp.matmul(post_in, weights.T)

        # Update compartment
        self.outputs.set(outputs)
        self.pre_out.set(pre_out)

    @compilable
    def reset(self):
        preVals = jnp.zeros((self.batch_size, self.shape[0]))
        postVals = jnp.zeros((self.batch_size, self.shape[1]))
        
        # BUG: the self.inputs here does not have the targeted field
        # NOTE: Quick workaround is to check if targeted is in the input or not
        hasattr(self.inputs, "targeted") and not self.inputs.targeted and self.inputs.set(preVals)
        self.outputs.set(postVals)
        self.post_in.set(postVals)
        self.pre_out.set(preVals)

    @classmethod
    def help(cls): ## component help function
        properties = {
            "synapse_type": "PatchedSynapse - performs a synaptic transformation "
                            "of inputs to produce  output signals (e.g., a "
                            "scaled linear multivariate transformation)"
        }
        compartment_props = {
            "inputs":
                {"inputs": "Takes in external input signal values",
                 "post_in": "Takes in external input signal values"},
            "states":
                {"weights": "Synapse efficacy/strength parameter values",
                 "biases": "Base-rate/bias parameter values",
                 "key": "JAX PRNG key"},
            "outputs":
                {"outputs": "Output of synaptic transformation",
                 "pre_out": "Output of synaptic transformation"},
        }
        hyperparams = {
            "shape": "Overall shape of synaptic weight value matrix; number inputs x number outputs",
            "n_sub_models": "The number of submodels (dense synaptic cables) in each layer",
            "stride_shape": "Stride shape of overlapping synaptic weight value matrix",
            "batch_size": "Batch size dimension of this component",
            "weight_init": "Initialization conditions for synaptic weight (W) values",
            "bias_init": "Initialization conditions for bias/base-rate (b) values",
            "resist_scale": "Resistance level scaling factor (Rscale); applied to output of transformation",
            "w_masks": "weight mask matrix",
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
        Wab = PatchedSynapse("Wab", (9, 30), 3)
    print(Wab)
    plt.imshow(Wab.weights.get(), cmap='gray')
    plt.show()






