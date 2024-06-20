from jax import random, numpy as jnp, jit
from ngclearn import resolver, Component, Compartment
from .deconvSynapse import DeconvSynapse
from ngclearn.components.synapses.convolution.ngcconv import (deconv2d, _calc_dX_deconv,
                                                              _calc_dK_deconv, calc_dX_deconv,
                                                              calc_dK_deconv)

class TraceSTDPDeconvSynapse(DeconvSynapse): ## trace-based STDP deconvolutional cable
    """
    A synaptic deconvolutional (transposed convolutional) cable that adjusts its
    filter efficacies via a trace-based form of spike-timing-dependent plasticity (STDP).

    | --- Synapse Compartments: ---
    | inputs - input (takes in external signals)
    | outputs - output signal (transformation induced by filters)
    | filters - current value matrix of synaptic filter efficacies
    | biases - current value vector of synaptic bias values
    | key - JAX RNG key
    | --- Synaptic Plasticity Compartments: ---
    | preSpike - pre-synaptic spike to drive 1st term of STDP update (takes in external signals)
    | postSpike - post-synaptic spike to drive 2nd term of STDP update (takes in external signals)
    | preTrace - pre-synaptic trace value to drive 1st term of STDP update (takes in external signals)
    | postTrace - post-synaptic trace value to drive 2nd term of STDP update (takes in external signals)
    | dWeights - delta tensor containing changes to be applied to synaptic efficacies
    | dInputs - delta tensor containing back-transmitted signal values ("backpropagating pulse")

    Args:
        name: the string name of this cell

        x_shape: dimension of input signal (assuming a square input)

        shape: tuple specifying shape of this synaptic cable (usually a 4-tuple
            with number `filter height x filter width x input channels x number output channels`);
            note that currently filters/kernels are assumed to be square
            (kernel.width = kernel.height)

        A_plus: strength of long-term potentiation (LTP)

        A_minus: strength of long-term depression (LTD)

        eta: global learning rate (default: 0)

        pretrace_target: controls degree of pre-synaptic disconnect, i.e., amount of decay
                 (higher -> lower synaptic values)

        filter_init: a kernel to drive initialization of this synaptic cable's
            filter values

        bias_init: kernel to drive initialization of bias/base-rate values
            (Default: None, which turns off/disables biases)

        stride: length/size of stride

        padding: pre-operator padding to use -- "VALID" (none), "SAME"

        resist_scale: a fixed (resistance) scaling factor to apply to synaptic
            transform (Default: 1.), i.e., yields: out = ((K @ in) * resist_scale) + b
            where `@` denotes convolution

        w_bound: maximum weight to softly bound this cable's value matrix to; if
            set to 0, then no synaptic value bounding will be applied

        w_decay: degree to which (L2) synaptic weight decay is applied to the
            computed STDP adjustment (Default: 0)

        batch_size: batch size dimension of this component
    """

    # Define Functions
    def __init__(self, name, shape, x_shape, A_plus, A_minus, eta=0.,
                 pretrace_target=0., filter_init=None, stride=1, padding=None,
                 resist_scale=1., w_bound=0., w_decay=0., batch_size=1,
                 **kwargs):
        super().__init__(name, shape, x_shape=x_shape, filter_init=filter_init,
                         bias_init=None, resist_scale=resist_scale, stride=stride,
                         padding=padding, batch_size=batch_size, **kwargs)

        self.eta = eta
        self.w_bounds = w_bound  ## soft weight constraint
        self.w_decay = w_decay  ## synaptic decay
        self.eta = eta  ## global learning rate governing plasticity
        self.pretrace_target = pretrace_target  ## target (pre-synaptic) trace activity value # 0.7
        self.Aplus = A_plus  ## LTP strength
        self.Aminus = A_minus  ## LTD strength

        ######################### set up compartments ##########################
        ## Compartment setup and shape computation
        self.dWeights = Compartment(self.weights.value * 0)
        self.dInputs = Compartment(jnp.zeros(self.in_shape))
        self.preSpike = Compartment(jnp.zeros(self.in_shape))
        self.preTrace = Compartment(jnp.zeros(self.in_shape))
        self.postSpike = Compartment(jnp.zeros(self.out_shape))
        self.postTrace = Compartment(jnp.zeros(self.out_shape))

        ########################################################################
        ## Shape error correction -- do shape correction inference (for local updates)
        self._init(self.batch_size, self.x_size, self.shape, self.stride,
                   self.padding, self.pad_args, self.weights)
        ########################################################################

    def _init(self, batch_size, x_size, shape, stride, padding, pad_args,
              weights):
        k_size, k_size, n_in_chan, n_out_chan = shape
        _x = jnp.zeros((batch_size, x_size, x_size, n_in_chan))
        _d = deconv2d(_x, self.weights.value, stride_size=self.stride,
                      padding=self.padding) * 0
        _dK = _calc_dK_deconv(_x, _d, stride_size=self.stride, out_size=k_size)
        ## get filter update correction
        dx = _dK.shape[0] - self.weights.value.shape[0]
        dy = _dK.shape[1] - self.weights.value.shape[1]
        self.delta_shape = (abs(dx), abs(dy))

        ## get input update correction
        _dx = _calc_dX_deconv(self.weights.value, _d, stride_size=self.stride,
                       padding=self.padding)
        dx = (_dx.shape[1] - _x.shape[1])  # abs()
        dy = (_dx.shape[2] - _x.shape[2])
        self.x_delta_shape = (dx, dy)

    @staticmethod
    def _evolve(eta, pretrace_target, Aplus, Aminus, w_decay, w_bounds,
                shape, stride, padding, delta_shape, preSpike, preTrace, postSpike,
                postTrace, weights):
        k_size, k_size, n_in_chan, n_out_chan = shape
        ## calc dFilters
        dW_ltp = calc_dK_deconv(preTrace - pretrace_target, postSpike * Aplus,
                                delta_shape=delta_shape, stride_size=stride,
                                out_size=k_size, padding=padding)
        dW_ltd = -calc_dK_deconv(preSpike, postTrace * Aminus,
                                 delta_shape=delta_shape, stride_size=stride,
                                 out_size=k_size, padding=padding)
        dWeights = (dW_ltp + dW_ltd)
        if w_decay > 0.: ## apply synaptic decay
            weights = weights + dWeights * eta - weights * w_decay  ## conduct decayed STDP-ascent
        else:
            weights = weights + dWeights * eta ## conduct STDP-ascent
        ## Apply any enforced filter constraints
        if w_bounds > 0.:
            ## enforce non-negativity
            eps = 0.01  # 0.001
            weights = jnp.clip(weights, eps, w_bounds - eps)
        return weights, dWeights

    @resolver(_evolve)
    def evolve(self, weights, dWeights):
        self.weights.set(weights)
        self.dWeights.set(dWeights)

    @staticmethod
    def _backtransmit(stride, padding, x_delta_shape, preSpike, postSpike,
                      weights):  ## action-backpropagating routine
        ## calc dInputs
        dInputs = calc_dX_deconv(weights, postSpike, delta_shape=x_delta_shape,
                                 stride_size=stride, padding=padding)
        return dInputs

    @resolver(_backtransmit)
    def backtransmit(self, dInputs):
        self.dInputs.set(dInputs)

    @staticmethod
    def _reset(in_shape, out_shape):
        preVals = jnp.zeros(in_shape)
        postVals = jnp.zeros(out_shape)
        inputs = preVals
        outputs = postVals
        preSpike = preVals
        postSpike = postVals
        preTrace = preVals
        postTrace = postVals
        return inputs, outputs, preSpike, postSpike, preTrace, postTrace

    @resolver(_reset)
    def reset(self, inputs, outputs, preSpike, postSpike, preTrace, postTrace):
        self.inputs.set(inputs)
        self.outputs.set(outputs)
        self.preSpike.set(preSpike)
        self.postSpike.set(postSpike)
        self.preTrace.set(preTrace)
        self.postTrace.set(postTrace)

    def help(self): ## component help function
        properties = {
            "synapse_type": "TraceSTDPDeconvSynapse - performs a synaptic deconvolution "
                            "(@.T) of inputs to produce output signals; synaptic "
                            "filters are adjusted via trace-based spike-timing-dependent "
                            "plasticity (STDP)"
        }
        compartment_props = {
            "input_compartments":
                {"inputs": "Takes in external input signal values",
                 "key": "JAX RNG key",
                 "preSpike": "Pre-synaptic spike compartment value/term for STDP (s_j)",
                 "postSpike": "Post-synaptic spike compartment value/term for STDP (s_i)",
                 "preTrace": "Pre-synaptic trace value term for STDP (z_j)",
                 "postTrace": "Post-synaptic trace value term for STDP (z_i)"},
            "parameter_compartments":
                {"filters": "Synaptic filter parameter values",
                 "biases": "Base-rate/bias parameter values"},
            "output_compartments":
                {"outputs": "Output of synaptic/filter transformation",
                 "dWeights": "Synaptic filter value adjustment 4D-tensor produced at time t",
                 "dInputs": "Tensor containing back-transmitted signal values; backpropagating pulse"},
        }
        hyperparams = {
            "shape": "Shape of synaptic filter value matrix; `kernel width` x `kernel height` "
                     "x `number input channels` x `number output channels`",
            "x_shape": "Shape of any single incoming/input feature map",
            "weight_init": "Initialization conditions for synaptic filter (K) values",
            "bias_init": "Initialization conditions for bias/base-rate (b) values",
            "resist_scale": "Resistance level output scaling factor (R)",
            "stride": "length / size of stride",
            "padding": "pre-operator padding to use, i.e., `VALID` `SAME`",
            "A_plus": "Strength of long-term potentiation (LTP)",
            "A_minus": "Strength of long-term depression (LTD)",
            "eta": "Global learning rate (multiplier beyond A_plus and A_minus)",
            "preTrace_target": "Pre-synaptic disconnecting/decay factor (x_tar)",
            "w_decay": "Synaptic filter decay term",
            "w_bound": "Soft synaptic bound applied to filters post-update"
        }
        info = {self.name: properties,
                "compartments": compartment_props,
                "dynamics": "outputs = [K @.T inputs] * R + b; "
                            "dW_{ij}/dt = A_plus * (z_j - x_tar) * s_i - A_minus * s_j * z_i",
                "hyperparameters": hyperparams}
        return info
