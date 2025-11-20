from jax import random, numpy as jnp, jit
from ngclearn import compilable #from ngcsimlib.parser import compilable
from ngclearn import Compartment #from ngcsimlib.compartment import Compartment
from ngclearn.components.synapses.convolution.convSynapse import ConvSynapse

from ngclearn.components.synapses.convolution.ngcconv import (_conv_same_transpose_padding,
                                                              _conv_valid_transpose_padding)
from ngclearn.components.synapses.convolution.ngcconv import (conv2d, _calc_dX_conv,
                                                              _calc_dK_conv, calc_dX_conv,
                                                              calc_dK_conv)

class TraceSTDPConvSynapse(ConvSynapse): ## trace-based STDP convolutional cable
    """
    A specialized synaptic convolutional cable that adjusts its filter efficacies via a trace-based form of
    spike-timing-dependent plasticity (STDP).

    | --- Synapse Compartments: ---
    | inputs - input (takes in external signals)
    | outputs - output signals (transformation induced by filters)
    | filters - current value matrix of synaptic filter efficacies
    | biases - current value vector of synaptic bias values
    | eta - learning rate global scale
    | key - JAX PRNG key
    | --- Synaptic Plasticity Compartments: ---
    | preSpike - pre-synaptic spike to drive 1st term of STDP update (takes in external signals)
    | postSpike - post-synaptic spike to drive 2nd term of STDP update (takes in external signals)
    | preTrace - pre-synaptic trace value to drive 1st term of STDP update (takes in external signals)
    | postTrace - post-synaptic trace value to drive 2nd term of STDP update (takes in external signals)
    | dWeights - delta tensor containing changes to be applied to synaptic filter efficacies
    | dInputs - delta tensor containing back-transmitted signal values ("backpropagating pulse")

    Args:
        name: the string name of this cell

        x_shape: 2d shape of input map signal (component currently assumess a square input maps)

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

    def __init__(
            self, name, shape, x_shape, A_plus, A_minus, eta=0., pretrace_target=0., filter_init=None, stride=1,
            padding=None, resist_scale=1., w_bound=0., w_decay=0., batch_size=1, **kwargs
    ):
        super().__init__(
            name, shape, x_shape=x_shape, filter_init=filter_init, bias_init=None, resist_scale=resist_scale,
            stride=stride, padding=padding, batch_size=batch_size, **kwargs
        )

        self.eta = eta
        self.w_bound = w_bound ## soft weight constraint
        self.w_decay = w_decay  ## synaptic decay
        self.eta = eta  ## global learning rate governing plasticity
        self.pretrace_target = pretrace_target  ## target (pre-synaptic) trace activity value # 0.7
        self.Aplus = A_plus  ## LTP strength
        self.Aminus = A_minus  ## LTD strength

        ######################### set up compartments ##########################
        ## Compartment setup and shape computation
        self.dWeights = Compartment(self.weights.get() * 0)
        self.dInputs = Compartment(jnp.zeros(self.in_shape))
        self.preSpike = Compartment(jnp.zeros(self.in_shape))
        self.preTrace = Compartment(jnp.zeros(self.in_shape))
        self.postSpike = Compartment(jnp.zeros(self.out_shape))
        self.postTrace = Compartment(jnp.zeros(self.out_shape))
        self.eta = Compartment(jnp.ones((1, 1)) * eta) ## global learning rate

        ########################################################################
        ## Shape error correction -- do shape correction inference for local updates
        self._init(self.batch_size, self.x_size, self.shape, self.stride,
                   self.padding, self.pad_args, self.weights)
        k_size, k_size, n_in_chan, n_out_chan = self.shape
        if padding == "SAME":
            self.antiPad = _conv_same_transpose_padding(
                self.postSpike.get().shape[1],
                self.x_size, k_size, stride)
        elif padding == "VALID":
            self.antiPad = _conv_valid_transpose_padding(
                self.postSpike.get().shape[1],
                self.x_size, k_size, stride)
        ########################################################################

    def _init(self, batch_size, x_size, shape, stride, padding, pad_args, weights):
        k_size, k_size, n_in_chan, n_out_chan = shape
        _x = jnp.zeros((batch_size, x_size, x_size, n_in_chan))
        _d = conv2d(_x, weights.get(), stride_size=stride, padding=padding) * 0
        _dK = _calc_dK_conv(_x, _d, stride_size=stride, padding=pad_args)
        ## get filter update correction
        dx = _dK.shape[0] - weights.get().shape[0]
        dy = _dK.shape[1] - weights.get().shape[1]
        #self.delta_shape = (dx, dy)
        self.delta_shape = (max(dx, 0), max(dy, 0))
        ## get input update correction
        _dx = _calc_dX_conv(weights.get(), _d, stride_size=stride,
                            anti_padding=pad_args)
        dx = (_dx.shape[1] - _x.shape[1])
        dy = (_dx.shape[2] - _x.shape[2])
        self.x_delta_shape = (dx, dy)

    def _compute_update(self):
        ## Compute long-term potentiation to filters
        dW_ltp = calc_dK_conv(
            self.preTrace.get() - self.pretrace_target, self.postSpike.get() * self.Aplus, delta_shape=self.delta_shape,
            stride_size=self.stride, padding=self.pad_args
        )
        ## Compute long-term depression to filters
        dW_ltd = -calc_dK_conv(
            self.preSpike.get(), self.postTrace.get() * self.Aminus, delta_shape=self.delta_shape,
            stride_size=self.stride, padding=self.pad_args
        )
        dWeights = (dW_ltp + dW_ltd)
        return dWeights

    @compilable
    def evolve(self):
        dWeights = self._compute_update()
        if self.w_decay > 0.:  ## apply synaptic decay
            weights = self.weights.get() + dWeights * self.eta - self.weights.get() * self.w_decay  ## conduct decayed STDP-ascent
        else:
            weights = self.weights.get() + dWeights * self.eta  ## conduct STDP-ascent
        ## Apply any enforced filter constraints
        if self.w_bound > 0.: ## enforce non-negativity
            eps = 0.01  # 0.001
            weights = jnp.clip(weights, eps, self.w_bound - eps)

        self.weights.set(weights)
        self.dWeights.set(dWeights)

    @compilable
    def backtransmit(self): ## action-backpropagating co-routine
        ## calc dInputs - adjustment w.r.t. input signal
        k_size, k_size, n_in_chan, n_out_chan = self.shape
        # antiPad = None
        # if padding == "SAME":
        #     antiPad = _conv_same_transpose_padding(postSpike.shape[1], x_size,
        #                                            k_size, stride)
        # elif padding == "VALID":
        #     antiPad = _conv_valid_transpose_padding(postSpike.shape[1], x_size,
        #                                             k_size, stride)
        dInputs = calc_dX_conv(
            self.weights.get(), self.postSpike.get(), delta_shape=self.x_delta_shape, stride_size=self.stride,
            anti_padding=self.antiPad
        )
        self.dInputs.set(dInputs)

    @compilable
    def reset(self):  # in_shape, out_shape):
        preVals = jnp.zeros(self.in_shape.get())
        postVals = jnp.zeros(self.out_shape.get())
        self.inputs.set(preVals)
        self.outputs.set(postVals)
        self.preSpike.set(preVals)
        self.postSpike.set(postVals)
        self.preTrace.set(preVals)
        self.postTrace.set(postVals)

    @classmethod
    def help(cls): ## component help function
        properties = {
            "synapse_type": "TraceSTDPConvSynapse - performs a synaptic convolution "
                            "(@) of inputs  to produce output signals; synaptic "
                            "filters are adjusted via trace-based spike-timing-dependent "
                            "plasticity (STDP)"
        }
        compartment_props = {
            "inputs":
                {"inputs": "Takes in external input signal values",
                 "preSpike": "Pre-synaptic spike compartment value/term for STDP (s_j)",
                 "postSpike": "Post-synaptic spike compartment value/term for STDP (s_i)",
                 "preTrace": "Pre-synaptic trace value term for STDP (z_j)",
                 "postTrace": "Post-synaptic trace value term for STDP (z_i)"},
            "states":
                {"filters": "Synaptic filter parameter values",
                 "biases": "Base-rate/bias parameter values",
                 "eta": "Global learning rate (multiplier beyond A_plus and A_minus)",
                 "key": "JAX PRNG key"},
            "analytics":
                {"dWeights": "Synaptic filter value adjustment 4D-tensor produced at time t",
                 "dInputs": "Tensor containing back-transmitted signal values; backpropagating pulse"},
            "outputs":
                {"outputs": "Output of synaptic/filter transformation"},
        }
        hyperparams = {
            "shape": "Shape of synaptic filter value matrix; `kernel width` x `kernel height` "
                     "x `number input channels` x `number output channels`",
            "x_shape": "Shape of any single incoming/input feature map",
            "filter_init": "Initialization conditions for synaptic filter (K) values",
            "resist_scale": "Resistance level output scaling factor (R)",
            "stride": "length / size of stride",
            "padding": "pre-operator padding to use, i.e., `VALID` `SAME`",
            "A_plus": "Strength of long-term potentiation (LTP)",
            "A_minus": "Strength of long-term depression (LTD)",
            "eta": "Global learning rate initial condition",
            "pretrace_target": "Pre-synaptic disconnecting/decay factor (x_tar)",
            "w_decay": "Synaptic filter decay term",
            "w_bound": "Soft synaptic bound applied to filters post-update"
        }
        info = {cls.__name__: properties,
                "compartments": compartment_props,
                "dynamics": "outputs = [K @ inputs] * R + b; "
                            "dW_{ij}/dt = A_plus * (z_j - x_tar) * s_i - A_minus * s_j * z_i",
                "hyperparameters": hyperparams}
        return info

