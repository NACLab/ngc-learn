from jax import random, numpy as jnp, jit
from ngclearn import compilable #from ngcsimlib.parser import compilable
from ngclearn import Compartment #from ngcsimlib.compartment import Compartment
from ngclearn.components.synapses.convolution.deconvSynapse import DeconvSynapse

from ngclearn.components.synapses.convolution.ngcconv import (deconv2d, _calc_dX_deconv,
                                                              _calc_dK_deconv, calc_dX_deconv,
                                                              calc_dK_deconv)
from ngclearn.utils.optim import get_opt_init_fn, get_opt_step_fn

class HebbianDeconvSynapse(DeconvSynapse): ## Hebbian-evolved deconvolutional cable
    """
    A specialized synaptic deconvolutional (transposed convolutional) cable that adjusts its efficacies via a
    two-factor Hebbian adjustment rule.

    | --- Synapse Compartments: ---
    | inputs - input (takes in external signals)
    | outputs - output signals (transformation induced by filters)
    | filters - current value matrix of synaptic filter efficacies
    | biases - current value vector of synaptic bias values
    | key - JAX PRNG key
    | --- Synaptic Plasticity Compartments: ---
    | pre - pre-synaptic signal to drive first term of Hebbian update (takes in external signals)
    | post - post-synaptic signal to drive 2nd term of Hebbian update (takes in external signals)
    | dWeights - delta tensor containing changes to be applied to synaptic filter efficacies
    | dBiases - delta tensor containing changes to be applied to bias values
    | dInputs - delta tensor containing back-transmitted signal values ("backpropagating pulse")
    | opt_params - locally-embedded optimizer statisticis (e.g., Adam 1st/2nd moments if adam is used)

    Args:
        name: the string name of this cell

        x_shape: dimension of input signal (assuming a square input)

        shape: tuple specifying shape of this synaptic cable (usually a 4-tuple
            with number `filter height x filter width x input channels x number output channels`);
            note that currently filters/kernels are assumed to be square
            (kernel.width = kernel.height)

        eta: global learning rate (default: 0)

        filter_init: a kernel to drive initialization of this synaptic cable's
            filter values

        bias_init: kernel to drive initialization of bias/base-rate values
            (Default: None, which turns off/disables biases)

        stride: length/size of stride

        padding: pre-operator padding to use -- "VALID" (none), "SAME"

        resist_scale: a fixed (resistance) scaling factor to apply to synaptic
            transform (Default: 1.), i.e., yields: out = ((W @.T Rscale) * in) + b
            where `@.T` denotes deconvolution

         w_bound: maximum weight to softly bound this cable's value matrix to; if
            set to 0, then no synaptic value bounding will be applied

        is_nonnegative: enforce that synaptic efficacies are always non-negative
            after each synaptic update (if False, no constraint will be applied)

        w_decay: degree to which (L2) synaptic weight decay is applied to the
            computed Hebbian adjustment (Default: 0); note that decay is not
            applied to any configured biases

        sign_value: multiplicative factor to apply to final synaptic update before
            it is applied to synapses; this is useful if gradient descent style
            optimization is required (as Hebbian rules typically yield
            adjustments for ascent)

        optim_type: optimization scheme to physically alter synaptic values
            once an update is computed (Default: "sgd"); supported schemes
            include "sgd" and "adam"

            :Note: technically, if "sgd" or "adam" is used but `signVal = 1`,
                then the ascent form of each rule is employed (signVal = -1) or
                a negative learning rate will mean a descent form of the
                `optim_scheme` is being employed

        batch_size: batch size dimension of this component
    """

    def __init__(
            self, name, shape, x_shape, eta=0., filter_init=None, bias_init=None, stride=1, padding=None,
            resist_scale=1., w_bound=0., is_nonnegative=False, w_decay=0., sign_value=1., optim_type="sgd",
            batch_size=1, **kwargs
    ):
        super().__init__(
            name, shape, x_shape=x_shape, filter_init=filter_init, bias_init=bias_init, resist_scale=resist_scale,
            stride=stride, padding=padding, batch_size=batch_size, **kwargs
        )

        self.eta = eta
        self.w_bounds = w_bound
        self.w_decay = w_decay  ## synaptic decay
        self.is_nonnegative = is_nonnegative
        self.sign_value = sign_value
        ## optimization / adjustment properties (given learning dynamics above)
        self.opt = get_opt_step_fn(optim_type, eta=self.eta)

        self.dWeights = Compartment(self.weights.get() * 0)
        self.dInputs = Compartment(jnp.zeros(self.in_shape))
        self.pre = Compartment(jnp.zeros(self.in_shape))
        self.post = Compartment(jnp.zeros(self.out_shape))
        self.dBiases = Compartment(self.biases.get() * 0)

        ########################################################################
        ## Shape error correction -- do shape correction inference (for local updates)
        self._init(self.batch_size, self.x_size, self.shape, self.stride,
                   self.padding, self.pad_args, self.weights)
        ########################################################################

        ## set up outer optimization compartments
        self.opt_params = Compartment(
            get_opt_init_fn(optim_type)([self.weights.get(), self.biases.get()] if bias_init else [self.weights.get()])
        )

    def _init(self, batch_size, x_size, shape, stride, padding, pad_args, weights):
        k_size, k_size, n_in_chan, n_out_chan = shape
        _x = jnp.zeros((batch_size, x_size, x_size, n_in_chan))
        _d = deconv2d(_x, self.weights.get(), stride_size=self.stride,
                      padding=self.padding) * 0
        _dK = _calc_dK_deconv(_x, _d, stride_size=self.stride, out_size=k_size)
        ## get filter update correction
        dx = _dK.shape[0] - self.weights.get().shape[0]
        dy = _dK.shape[1] - self.weights.get().shape[1]
        self.delta_shape = (abs(dx), abs(dy))

        ## get input update correction
        _dx = _calc_dX_deconv(self.weights.get(), _d, stride_size=self.stride,
                       padding=self.padding)
        dx = (_dx.shape[1] - _x.shape[1])  # abs()
        dy = (_dx.shape[2] - _x.shape[2])
        self.x_delta_shape = (dx, dy)

    def _compute_update(self):
        k_size, k_size, n_in_chan, n_out_chan = self.shape
        ## compute adjustment to filters
        dWeights = calc_dK_deconv(
            self.pre.get(), self.post.get(), delta_shape=self.delta_shape, stride_size=self.stride, out_size=k_size,
            padding=self.padding
        )
        dWeights = dWeights * self.sign_value
        if self.w_decay > 0.:  ## apply synaptic decay
            dWeights = dWeights - self.weights.get() * self.w_decay
        ## compute adjustment to base-rates (if applicable)
        dBiases = 0.  # jnp.zeros((1,1))
        if self.bias_init != None:
            dBiases = jnp.sum(self.post.get(), axis=0, keepdims=True) * self.sign_value
        return dWeights, dBiases

    @compilable
    def evolve(self):
        dWeights, dBiases = self._compute_update()
        if self.bias_init != None:
            opt_params, [weights, biases] = self.opt(self.opt_params.get(), [self.weights.get(), self.biases.get()], [dWeights, dBiases])
        else: ## ignore dBiases since no biases configured
            opt_params, [weights] = self.opt(self.opt_params.get(), [self.weights.get()], [dWeights])
            biases = None
        ## apply any enforced filter constraints
        if self.w_bounds > 0.:
            if self.is_nonnegative:
                weights = jnp.clip(weights, 0., self.w_bounds)
            else:
                weights = jnp.clip(weights, -self.w_bounds, self.w_bounds)

        self.opt_params.set(opt_params)
        self.weights.set(weights)
        self.biases.set(biases)
        self.dWeights.set(dWeights)
        self.dBiases.set(dBiases)

    @compilable
    def backtransmit(self): ## action-backpropagating co-routine
        ## calc dInputs
        dInputs = calc_dX_deconv(
            self.weights.get(), self.post.get(), delta_shape=self.x_delta_shape, stride_size=self.stride,
            padding=self.padding
        )
        ## flip sign of back-transmitted signal (if applicable)
        dInputs = dInputs * self.sign_value
        self.dInputs.set(dInputs)

    @compilable
    def reset(self): #in_shape, out_shape):
        preVals = jnp.zeros(self.in_shape.get())
        postVals = jnp.zeros(self.out_shape.get())
        self.inputs.set(preVals)
        self.outputs.set(postVals)
        self.pre.set(preVals)
        self.post.set(postVals)
        self.dInputs.set(preVals)

    @classmethod
    def help(cls): ## component help function
        properties = {
            "synapse_type": "DeconvSynapse - performs a synaptic deconvolution "
                            "(@.T) of inputs to produce output signals; synaptic "
                            "filters are adjusted via two-term/factor Hebbian "
                            "adjustment"
        }
        compartment_props = {
            "inputs":
                {"inputs": "Takes in external input signal values",
                 "pre": "Pre-synaptic statistic for Hebb rule (z_j)",
                 "post": "Post-synaptic statistic for Hebb rule (z_i)"},
            "states":
                {"filters": "Synaptic filter parameter values",
                 "biases": "Base-rate/bias parameter values",
                 "key": "JAX PRNG key"},
            "analytics":
                {"dWeights": "Synaptic filter value adjustment 4D-tensor produced at time t",
                 "dBiases": "Synaptic bias/base-rate value adjustment 3D-tensor produced at time t"},
            "outputs":
                {"outputs": "Output of synaptic/filter transformation",
                 "dInputs": "Tensor containing back-transmitted signal values; backpropagating pulse"},
        }
        hyperparams = {
            "shape": "Shape of synaptic filter value matrix; `kernel width` x `kernel height` "
                     "x `number input channels` x `number output channels`",
            "x_shape": "Shape of any single incoming/input feature map",
            "filter_init": "Initialization conditions for synaptic filter (K) values",
            "bias_init": "Initialization conditions for bias/base-rate (b) values",
            "resist_scale": "Resistance level output scaling factor (R)",
            "stride": "length / size of stride",
            "padding": "pre-operator padding to use, i.e., `VALID` `SAME`",
            "is_nonnegative": "Should filters be constrained to be non-negative post-updates?",
            "sign_value": "Scalar `flipping` constant -- changes direction to Hebbian descent if < 0",
            "eta": "Global (fixed) learning rate",
            "w_bound": "Soft synaptic bound applied to filters post-update",
            "w_decay": "Synaptic filter decay term",
            "optim_type": "Optimization scheme to used for adjusting synapses"
        }
        info = {cls.__name__: properties,
                "compartments": compartment_props,
                "dynamics": "outputs = [K @.T inputs] * R + b; "
                            "dW_{ij}/dt = eta * [(z_j * q_pre) * (z_i * q_post)] - W_{ij} * w_decay",
                "hyperparameters": hyperparams}
        return info
