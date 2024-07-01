from jax import random, numpy as jnp, jit
from ngclearn import resolver, Component, Compartment
from ngclearn.components.synapses import DenseSynapse
from ngclearn.utils import tensorstats

class GatedHebbianSynapse(DenseSynapse):

    # Define Functions
    def __init__(self, name, shape, eta=0., weight_init=None, bias_init=None,
                 w_bound=1., w_decay=0., alpha=0., p_conn=1., resist_scale=1.,
                 batch_size=1, **kwargs):
        super().__init__(name, shape, weight_init, bias_init, resist_scale,
                         p_conn, batch_size=batch_size, **kwargs)

        ## synaptic plasticity properties and characteristics
        self.shape = shape
        self.w_bound = w_bound
        self.w_decay = w_decay ## synaptic decay
        self.eta = eta
        self.alpha = alpha

        # compartments (state of the cell, parameters, will be updated through stateless calls)
        self.preVals = jnp.zeros((self.batch_size, shape[0]))
        self.postVals = jnp.zeros((self.batch_size, shape[1]))
        self.pre = Compartment(self.preVals)
        self.post = Compartment(self.postVals)
        self.preSpike = Compartment(self.preVals)
        self.postSpike = Compartment(self.postVals)
        self.dWeights = Compartment(jnp.zeros(shape))
        self.dBiases = Compartment(jnp.zeros(shape[1]))

    @staticmethod
    def _compute_update(alpha, pre, post, weights):
        ## calculate synaptic update values
        dW = jnp.matmul(pre.T, post)
        db = jnp.sum(post, axis=0, keepdims=True)
        if alpha > 0.: ## apply synaptic dependency weighting
            dW = dW * (alpha - jnp.abs(weights))
        return dW, db

    @staticmethod
    def _evolve(bias_init, eta, alpha, w_decay, w_bound, pre, post, preSpike,
                postSpike, weights, biases):
        ## calculate synaptic update values
        dWeights, dBiases = GatedHebbianSynapse._compute_update(alpha, pre, post, weights)
        weights = weights + dWeights * eta
        if w_decay > 0.:
            Wdec = jnp.matmul((1. - preSpike).T, postSpike) * w_decay
            weights = weights - Wdec
        if bias_init != None:
            biases = biases + dBiases * eta
        weights = jnp.clip(weights, 0., w_bound)
        return weights, biases, dWeights, dBiases

    @resolver(_evolve)
    def evolve(self, weights, biases, dWeights, dBiases):
        self.weights.set(weights)
        self.biases.set(biases)
        self.dWeights.set(dWeights)
        self.dBiases.set(dBiases)

    @staticmethod
    def _reset(batch_size, shape):
        preVals = jnp.zeros((batch_size, shape[0]))
        postVals = jnp.zeros((batch_size, shape[1]))
        return (
            preVals, # inputs
            postVals, # outputs
            preVals, # pre
            postVals, # post
            preVals,  # pre
            postVals,  # post
            jnp.zeros(shape), # dW
            jnp.zeros(shape[1]), # db
        )

    @resolver(_reset)
    def reset(self, inputs, outputs, pre, post, preSpike, postSpike, dWeights, dBiases):
        self.inputs.set(inputs)
        self.outputs.set(outputs)
        self.pre.set(pre)
        self.post.set(post)
        self.preSpike.set(preSpike)
        self.postSpike.set(postSpike)
        self.dWeights.set(dWeights)
        self.dBiases.set(dBiases)

    @classmethod
    def help(cls): ## component help function
        properties = {
            "synapse_type": "HebbianSynapse - performs an adaptable synaptic "
                            "transformation of inputs to produce output signals; "
                            "synapses are adjusted via two-term/factor Hebbian adjustment"
        }
        compartment_props = {
            "inputs":
                {"inputs": "Takes in external input signal values",
                 "pre": "Pre-synaptic statistic for Hebb rule (z_j)",
                 "post": "Post-synaptic statistic for Hebb rule (z_i)"},
            "states":
                {"weights": "Synapse efficacy/strength parameter values",
                 "biases": "Base-rate/bias parameter values",
                 "key": "JAX PRNG key"},
            "analytics":
                {"dWeights": "Synaptic weight value adjustment matrix produced at time t",
                 "dBiases": "Synaptic bias/base-rate value adjustment vector produced at time t"},
            "outputs":
                {"outputs": "Output of synaptic transformation"},
        }
        hyperparams = {
            "shape": "Shape of synaptic weight value matrix; number inputs x number outputs",
            "batch_size": "Batch size dimension of this component",
            "weight_init": "Initialization conditions for synaptic weight (W) values",
            "bias_init": "Initialization conditions for bias/base-rate (b) values",
            "resist_scale": "Resistance level scaling factor (applied to output of transformation)",
            "p_conn": "Probability of a connection existing (otherwise, it is masked to zero)",
            "is_nonnegative": "Should synapses be constrained to be non-negative post-updates?",
            "sign_value": "Scalar `flipping` constant -- changes direction to Hebbian descent if < 0",
            "eta": "Global (fixed) learning rate",
            "pre_wght": "Pre-synaptic weighting coefficient (q_pre)",
            "post_wght": "Post-synaptic weighting coefficient (q_post)",
            "w_bound": "Soft synaptic bound applied to synapses post-update",
            "w_decay": "Synaptic decay term",
            "optim_type": "Choice of optimizer to adjust synaptic weights"
        }
        info = {cls.__name__: properties,
                "compartments": compartment_props,
                "dynamics": "outputs = [(W * Rscale) * inputs] + b ;"
                            "dW_{ij}/dt = eta * [(z_j * q_pre) * (z_i * q_post)] - W_{ij} * w_decay",
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
