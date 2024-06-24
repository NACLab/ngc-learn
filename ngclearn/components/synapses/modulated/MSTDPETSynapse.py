from jax import random, numpy as jnp, jit
from ngclearn import resolver, Component, Compartment
from ngclearn.components.synapses.hebbian import TraceSTDPSynapse
from ngclearn.utils import tensorstats

class MSTDPETSynapse(TraceSTDPSynapse): # modulated trace-based STDP w/ eligility traces
    # Define Functions
    def __init__(self, name, shape, A_plus, A_minus, eta=1., mu=0.,
                 pretrace_target=0., elg_tau=0., elg_decay=1., update_scale=1.,
                 weight_init=None, resist_scale=1., p_conn=1., **kwargs):
        super().__init__(name,shape, A_plus, A_minus, eta=eta, mu=mu,
                         pretrace_target=pretrace_target, weight_init=weight_init,
                         resist_scale=resist_scale, p_conn=p_conn, **kwargs)
        ## MSTDP/MSTDP-ET meta-parameters
        self.elg_tau = elg_tau
        self.elg_decay = elg_decay
        self.update_scale = update_scale
        ## MSTDP/MSTDP-ET compartments
        self.modulator = Compartment(jnp.zeros((self.batch_size, 1)))
        self.eligiblity = Compartment(jnp.zeros(shape))

    @staticmethod
    def _evolve(dt, w_bound, preTrace_target, mu, Aplus, Aminus, elg_tau,
                elg_decay, update_scale, preSpike, postSpike, preTrace, postTrace,
                weights, eta, modulator, eligiblity):
        ## compute local synaptic update (via STDP)
        dW_dt = super()._compute_update(
            dt, w_bound, preTrace_target, mu, Aplus, Aminus,
            preSpike, postSpike, preTrace, postTrace, weights
        )
        if elg_tau > 0.: ## perform dynamics of M-STDP-ET
            ## update eligibility trace given current local update
            dElg_dt = -eligiblity * elg_decay + dW_dt * update_scale
            eligiblity = eligiblity + dElg_dt * dt/elg_tau
        else: ## recovers M-STDP
            eligiblity = dW_dt
        dWeights = eligiblity * modulator ## trace/update times modulatory signal (e.g., reward)

        if eta > 0.: ## perform physical adjustment of synapses
            ## do a gradient ascent update/shift
            weights = weights + dWeights * eta ## modulate update
            ## enforce non-negativity
            eps = 0.01 # 0.001
            weights = jnp.clip(weights, eps, w_bound - eps)  # jnp.abs(w_bound))
        return weights, dWeights

    @resolver(_evolve)
    def evolve(self, weights, dWeights):
        self.weights.set(weights)
        self.dWeights.set(dWeights)

    @staticmethod
    def _reset(batch_size, shape):
        preVals = jnp.zeros((batch_size, shape[0]))
        postVals = jnp.zeros((batch_size, shape[1]))
        inputs = preVals
        outputs = postVals
        preSpike = preVals
        postSpike = postVals
        preTrace = preVals
        postTrace = postVals
        dWeights = jnp.zeros(shape)
        return inputs, outputs, preSpike, postSpike, preTrace, postTrace, dWeights

    @resolver(_reset)
    def reset(self, inputs, outputs, preSpike, postSpike, preTrace, postTrace, dWeights):
        self.inputs.set(inputs)
        self.outputs.set(outputs)
        self.preSpike.set(preSpike)
        self.postSpike.set(postSpike)
        self.preTrace.set(preTrace)
        self.postTrace.set(postTrace)
        self.dWeights.set(dWeights)

    def help(self): ## component help function
        properties = {
            "synapse_type": "TraceSTDPSynapse - performs an adaptable synaptic "
                            "transformation of inputs to produce output signals; "
                            "synapses are adjusted with trace-based "
                            "spike-timing-dependent plasticity (STDP)"
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
                {"weights": "Synapse efficacy/strength parameter values",
                 "biases": "Base-rate/bias parameter values"},
            "output_compartments":
                {"outputs": "Output of synaptic transformation",
                 "dWeights": "Synaptic weight value adjustment matrix produced at time t"},
        }
        hyperparams = {
            "shape": "Shape of synaptic weight value matrix; number inputs x number outputs",
            "weight_init": "Initialization conditions for synaptic weight (W) values",
            "resist_scale": "Resistance level scaling factor (applied to output of transformation)",
            "p_conn": "Probability of a connection existing (otherwise, it is masked to zero)",
            "A_plus": "Strength of long-term potentiation (LTP)",
            "A_minus": "Strength of long-term depression (LTD)",
            "eta": "Global learning rate (multiplier beyond A_plus and A_minus)",
            "mu": "Power factor for STDP adjustment",
            "preTrace_target": "Pre-synaptic disconnecting/decay factor (x_tar)",
        }
        info = {self.name: properties,
                "compartments": compartment_props,
                "dynamics": "outputs = [(W * Rscale) * inputs] ;"
                            "dW_{ij}/dt = A_plus * (z_j - x_tar) * s_i - A_minus * s_j * z_i",
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
