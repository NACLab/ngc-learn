# %%

import matplotlib.pyplot as plt
from jax import random, numpy as jnp, jit
from ngclearn.components.jaxComponent import JaxComponent
from ngclearn.utils.distribution_generator import DistributionGenerator

from ngcsimlib.logger import info
from ngclearn import compilable #from ngcsimlib.parser import compilable
from ngclearn import Compartment #from ngcsimlib.compartment import Compartment
# from ngclearn.utils.weight_distribution import initialize_params


# def _create_multi_patch_synapses(key, shape, n_sub_models, sub_stride, weight_init):
#     sub_shape = (shape[0] // n_sub_models, shape[1] // n_sub_models)
#     di, dj = sub_shape
#     si, sj = sub_stride

#     weight_shape = ((n_sub_models * di) + 2 * si, (n_sub_models * dj) + 2 * sj)
#     #weights = initialize_params(key[2], {"dist": "constant", "value": 0.}, weight_shape, use_numpy=True)
#     large_weight_init = DistributionGenerator.constant(value=0.)
#     weights = large_weight_init(weight_shape, key[2])

#     for i in range(n_sub_models):
#         start_i = i * di
#         end_i = (i + 1) * di + 2 * si
#         start_j = i * dj
#         end_j = (i + 1) * dj + 2 * sj

#         shape_ = (end_i - start_i, end_j - start_j) # (di + 2 * si, dj + 2 * sj)

#         ## FIXME: this line below might be wonky...
#         weights.at[start_i: end_i, start_j: end_j].set( weight_init(shape_, key[2]) )
#         # weights[start_i : end_i,
#         #         start_j : end_j] = initialize_params(key[2], init_kernel=weight_init, shape=shape_, use_numpy=True)
#     if si != 0:
#         weights.at[:si,:].set(0.) ## FIXME: this setter line might be wonky...
#         weights.at[-si:,:].set(0.) ## FIXME: this setter line might be wonky...
#     if sj != 0:
#         weights.at[:,:sj].set(0.) ## FIXME: this setter line might be wonky...
#         weights.at[:, -sj:].set(0.) ## FIXME: this setter line might be wonky...

#     return weights

def _create_multi_patch_synapses(key, shape, n_sub_models, sub_stride, weight_init):
    sub_shape = (shape[0] // n_sub_models, shape[1] // n_sub_models)
    di, dj = sub_shape
    si, sj = sub_stride

    weight_shape = ((n_sub_models * di) + 2 * si, (n_sub_models * dj) + 2 * sj)
    # weights = initialize_params(key[2], {"dist": "constant", "value": 0.}, weight_shape, use_numpy=True)
    weights = DistributionGenerator.constant(value=0.)(weight_shape, key[2])

    for i in range(n_sub_models):
        start_i = i * di
        end_i = (i + 1) * di + 2 * si
        start_j = i * dj
        end_j = (i + 1) * dj + 2 * sj

        shape_ = (end_i - start_i, end_j - start_j) # (di + 2 * si, dj + 2 * sj)

        # weights[start_i : end_i,
        #         start_j : end_j] = initialize_params(key[2],
        #                                              init_kernel=weight_init,
        #                                              shape=shape_,
        #                                              use_numpy=True)
        weights = weights.at[start_i : end_i,
                start_j : end_j].set(weight_init(shape_, key[2]))
    if si!=0:
        weights = weights.at[:si,:].set(0.)
        weights = weights.at[-si:,:].set(0.)
    if sj!=0:
        weights = weights.at[:,:sj].set(0.)
        weights = weights.at[:, -sj:].set(0.)

    return weights


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

        block_mask: weight mask matrix

        pre_wght: pre-synaptic weighting factor (Default: 1.)

        post_wght: post-synaptic weighting factor (Default: 1.)

        resist_scale: a fixed scaling factor to apply to synaptic transform
            (Default: 1.), i.e., yields: out = ((W * Rscale) * in) + b

        p_conn: probability of a connection existing (default: 1.); setting
            this to < 1. will result in a sparser synaptic structure
    """

    def __init__(
            self, name, shape, n_sub_models=1, stride_shape=(0,0), block_mask=None, weight_init=None, bias_init=None,
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
            #self.weight_init = {"dist": "fan_in_gaussian"}
            self.weight_init = DistributionGenerator.fan_in_gaussian()

        weights = _create_multi_patch_synapses(
            key=subkeys, shape=shape, n_sub_models=self.n_sub_models, sub_stride=self.sub_stride,
            weight_init=self.weight_init
        )

        self.block_mask = jnp.where(weights!=0, 1, 0)
        self.sub_shape = (shape[0]//n_sub_models, shape[1]//n_sub_models)

        self.shape = weights.shape
        self.sub_shape = self.sub_shape[0]+(2*self.sub_stride[0]), self.sub_shape[1]+(2*self.sub_stride[1])

        if 0. < p_conn < 1.: ## only non-zero and <1 probs allowed
            mask = random.bernoulli(subkeys[1], p=p_conn, shape=self.shape)
            weights = weights * mask ## sparsify matrix

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
        self.biases = Compartment(self.bias_init((1, self.shape[1]), subkeys[2]) if bias_init else 0.0)
        #elf.biases = Compartment(initialize_params(subkeys[2], bias_init, (1, self.shape[1])) if bias_init else 0.0)

    @compilable
    def advance_state(self):
        # Get the variables
        inputs = self.inputs.get()
        weights = self.weights.get()
        biases = self.biases.get()

        outputs = (jnp.matmul(inputs, weights) * self.Rscale) + biases

        # Update compartment
        self.outputs.set(outputs)

    @compilable
    def reset(self):
        preVals = jnp.zeros((self.batch_size, self.shape[0]))
        postVals = jnp.zeros((self.batch_size, self.shape[1]))
        inputs = preVals
        outputs = postVals
        # BUG: the self.inputs here does not have the targeted field
        # NOTE: Quick workaround is to check if targeted is in the input or not
        hasattr(self.inputs, "targeted") and not self.inputs.targeted and self.inputs.set(inputs)
        self.outputs.set(outputs)

    @classmethod
    def help(cls): ## component help function
        properties = {
            "synapse_type": "PatchedSynapse - performs a synaptic transformation "
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
            "shape": "Overall shape of synaptic weight value matrix; number inputs x number outputs",
            "n_sub_models": "The number of submodels (dense synaptic cables) in each layer",
            "stride_shape": "Stride shape of overlapping synaptic weight value matrix",
            "batch_size": "Batch size dimension of this component",
            "weight_init": "Initialization conditions for synaptic weight (W) values",
            "bias_init": "Initialization conditions for bias/base-rate (b) values",
            "resist_scale": "Resistance level scaling factor (Rscale); applied to output of transformation",
            "block_mask": "weight mask matrix",
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

