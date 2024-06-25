from jax import random, numpy as jnp, jit
from ngclearn import resolver, Component, Compartment
from ngclearn.components.synapses.convolution import TraceSTDPConvSynapse
from ngclearn.utils import tensorstats

class MSTDPETConvSynapse(TraceSTDPConvSynapse): # modulated trace-based conv STDP w/ eligility traces
    """
    A synaptic convolutional cable that adjusts its
    filter efficacies through a form of modulated spike-timing-dependent
    plasticity (MSTDP) or modulated STDP with eligibility traces (MSTDP-ET).

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

        tau_elg: eligibility trace time constant (default: 0); must be >0,
            otherwise, the trace is disabled and this synapse evolves via M-STDP

        elg_decay: eligibility decay constant (default: 1)

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

    # Define Functions
    def __init__(self, name, shape, x_shape, A_plus, A_minus, eta=0.,
                 pretrace_target=0., tau_elg=0., elg_decay=1., filter_init=None,
                 stride=1, padding=None, resist_scale=1., w_bound=0., w_decay=0.,
                 batch_size=1, **kwargs):
        super().__init__(name, shape, x_shape=x_shape,
                         A_plus=A_plus, A_minus=A_minus, eta=eta,
                         pretrace_target=pretrace_target, w_bound=w_bound, w_decay=w_decay,
                         filter_init=filter_init, bias_init=None, resist_scale=resist_scale,
                         stride=stride, padding=padding, batch_size=batch_size, **kwargs)
        ## MSTDP/MSTDP-ET meta-parameters
        self.tau_elg = tau_elg
        self.elg_decay = elg_decay
        ## MSTDP/MSTDP-ET compartments
        self.modulator = Compartment(jnp.zeros((self.batch_size, 1)))
        self.eligibility = Compartment(jnp.zeros(shape))

    @staticmethod
    def _evolve(dt, pretrace_target, Aplus, Aminus, w_decay, w_bound,
                stride, pad_args, delta_shape, tau_elg, elg_decay, preSpike,
                preTrace, postSpike, postTrace, weights, eta, modulator, eligibility):
        ## compute local synaptic update (via STDP)
        dW_dt = TraceSTDPConvSynapse._compute_update(
            pretrace_target, Aplus, Aminus, stride, pad_args, delta_shape,
            preSpike, preTrace, postSpike, postTrace
        ) ## produce dW/dt (ODE for synaptic change dynamics)
        if tau_elg > 0.: ## perform dynamics of M-STDP-ET
            ## update eligibility trace given current local update
            eligibility = eligibility * jnp.exp(-dt / tau_elg) * elg_decay + dW_dt
        else: ## perform dynamics of M-STDP (no eligibility trace)
            eligibility = dW_dt
        ## Perform a trace/update times a modulatory signal (e.g., reward)
        dWeights = eligibility * modulator

        ## do a gradient ascent update/shift
        weights = weights + dWeights * eta ## modulate update
        if w_decay > 0.:
            weights = weights - weights * w_decay
        if w_bound > 0.: ## enforce non-negativity
            eps = 0.01
            weights = jnp.clip(weights, eps, w_bound - eps)  # jnp.abs(w_bound))
        return weights, dWeights, eligibility

    @resolver(_evolve)
    def evolve(self, weights, dWeights, eligibility):
        self.weights.set(weights)
        self.dWeights.set(dWeights)
        self.eligibility.set(eligibility)

    @staticmethod
    def _reset(in_shape, out_shape, shape):
        preVals = jnp.zeros(in_shape)
        postVals = jnp.zeros(out_shape)
        inputs = preVals
        outputs = postVals
        preSpike = preVals
        postSpike = postVals
        preTrace = preVals
        postTrace = postVals
        eligibility = jnp.zeros(shape)
        return (inputs, outputs, preSpike, postSpike, preTrace,
                postTrace, eligibility)

    @resolver(_reset)
    def reset(self, inputs, outputs, preSpike, postSpike, preTrace, postTrace,
              eligibility):
        self.inputs.set(inputs)
        self.outputs.set(outputs)
        self.preSpike.set(preSpike)
        self.postSpike.set(postSpike)
        self.preTrace.set(preTrace)
        self.postTrace.set(postTrace)
        self.eligibility.set(eligibility)

    @classmethod
    def help(cls): ## component help function
        properties = {
            "synapse_type": "MSTDPETConvSynapse - performs a synaptic convolution "
                            "(@) of inputs  to produce output signals; synaptic "
                            "filters are adjusted via a form of modulated "
                            "spike-timing-dependent plasticity (MSTDP) or "
                            "MSTDP w/ eligibility traces (MSTDP-ET)"
        }
        compartment_props = {
            "inputs":
                {"inputs": "Takes in external input signal values",
                 "preSpike": "Pre-synaptic spike compartment value/term for STDP (s_j)",
                 "postSpike": "Post-synaptic spike compartment value/term for STDP (s_i)",
                 "preTrace": "Pre-synaptic trace value term for STDP (z_j)",
                 "postTrace": "Post-synaptic trace value term for STDP (z_i)",
                 "modulator": "External modulatory signal values (e.g., reward values) (r)"},
            "states":
                {"filters": "Synaptic filter parameter values",
                 "biases": "Base-rate/bias parameter values",
                 "eligibility": "Current state of eligibility trace at time `t` (Elg)",
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
            "batch_size": "Batch size dimension of this component",
            "filter_init": "Initialization conditions for synaptic filter (K) values",
            "resist_scale": "Resistance level output scaling factor (R)",
            "stride": "length / size of stride",
            "padding": "pre-operator padding to use, i.e., `VALID` `SAME`",
            "A_plus": "Strength of long-term potentiation (LTP)",
            "A_minus": "Strength of long-term depression (LTD)",
            "eta": "Global learning rate initial condition",
            "preTrace_target": "Pre-synaptic disconnecting/decay factor (x_tar)",
            "w_decay": "Synaptic filter decay term",
            "w_bound": "Soft synaptic bound applied to filters post-update",
            "tau_elg": "Eligibility trace time constant",
            "elg_decay": "Eligibility decay factor"
        }
        info = {cls.__name__: properties,
                "compartments": compartment_props,
                "dynamics": "outputs = [K @ inputs] * R + b; "
                            "dW_{ij}/dt = Elg * r * eta; " 
                            "dElg/dt = -Elg * elg_decay + dW^{stdp}_{ij}/dt" 
                            "dW^{stdp}_{ij}/dt = A_plus * (z_j - x_tar) * s_i - A_minus * s_j * z_i",
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
