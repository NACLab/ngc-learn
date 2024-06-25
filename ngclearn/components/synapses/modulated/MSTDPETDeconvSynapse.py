from jax import random, numpy as jnp, jit
from ngclearn import resolver, Component, Compartment
from ngclearn.components.synapses.convolution import TraceSTDPDeconvSynapse

class TraceSTDPDeconvSynapse(TraceSTDPDeconvSynapse): ## modulated trace-based deconv STDP cable

    # Define Functions
    def __init__(self, name, shape, x_shape, A_plus, A_minus, eta=0.,
                 pretrace_target=0., tau_elg=0., elg_decay=1., filter_init=None,
                 stride=1, padding=None, resist_scale=1., w_bound=0., w_decay=0.,
                 batch_size=1, **kwargs):
        super().__init__(name, shape, x_shape=x_shape, A_plus=A_plus, A_minus=A_minus,
                         eta=eta, pretrace_target=pretrace_target, filter_init=filter_init,
                         bias_init=None, stride=stride, padding=padding,
                         resist_scale=resist_scale, w_bound=w_bound, w_decay=w_decay,
                         batch_size=batch_size,**kwargs)
        ## MSTDP/MSTDP-ET meta-parameters
        self.tau_elg = tau_elg
        self.elg_decay = elg_decay
        ## MSTDP/MSTDP-ET compartments
        self.modulator = Compartment(jnp.zeros((self.batch_size, 1)))
        self.eligibility = Compartment(jnp.zeros(shape))
        ########################################################################

    @staticmethod
    def _evolve(dt, pretrace_target, Aplus, Aminus, w_decay, w_bound, shape, stride,
                padding, delta_shape, tau_elg, elg_decay, preSpike, preTrace,
                postSpike, postTrace, weights, eta, modulator, eligibility):
        dW_dt = TraceSTDPDeconvSynapse._compute_update(
            pretrace_target, Aplus, Aminus, shape, stride, padding, delta_shape,
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
            "synapse_type": "MSTDPETDeconvSynapse - performs a synaptic deconvolution "
                            "(@.T) of inputs to produce output signals; synaptic "
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
            "filter_init": "Initialization conditions for synaptic filter (K) values",
            "bias_init": "Initialization conditions for bias/base-rate (b) values",
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
                "dynamics": "outputs = [K @.T inputs] * R + b; "
                            "dW_{ij}/dt = Elg * r * eta; " 
                            "dElg/dt = -Elg * elg_decay + dW^{stdp}_{ij}/dt" 
                            "dW_{ij}/dt = A_plus * (z_j - x_tar) * s_i - A_minus * s_j * z_i",
                "hyperparameters": hyperparams}
        return info
