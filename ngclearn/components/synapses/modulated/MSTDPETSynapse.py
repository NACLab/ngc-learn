from jax import random, numpy as jnp, jit
from ngclearn import resolver, Component, Compartment
from ngclearn.components.synapses.hebbian import TraceSTDPSynapse
from ngclearn.utils import tensorstats

class MSTDPETSynapse(TraceSTDPSynapse): # modulated trace-based STDP w/ eligility traces
    """
    A synaptic cable that adjusts its efficacies via trace-based form of
    three-factor learning, i.e., modulated spike-timing-dependent plasticity
    (M-STDP) or modulated STDP with eligibility traces (M-STDP-ET).

    | --- Synapse Compartments: ---
    | inputs - input (takes in external signals)
    | outputs - output signals (transformation induced by synapses)
    | weights - current value matrix of synaptic efficacies
    | modulator - external modulatory signal values (e.g., a reward value)
    | key - JAX PRNG key
    | --- Synaptic Plasticity Compartments: ---
    | preSpike - pre-synaptic spike to drive 1st term of STDP update (takes in external signals)
    | postSpike - post-synaptic spike to drive 2nd term of STDP update (takes in external signals)
    | preTrace - pre-synaptic trace value to drive 1st term of STDP update (takes in external signals)
    | postTrace - post-synaptic trace value to drive 2nd term of STDP update (takes in external signals)
    | dWeights - current delta matrix containing changes to be applied to synaptic efficacies
    | eligibility - current state of eligibility trace
    | eta - global learning rate (multiplier beyond A_plus and A_minus)

    | References:
    | Morrison, Abigail, Ad Aertsen, and Markus Diesmann. "Spike-timing-dependent
    | plasticity in balanced random networks." Neural computation 19.6 (2007): 1437-1467.
    |
    | Bi, Guo-qiang, and Mu-ming Poo. "Synaptic modification by correlated
    | activity: Hebb's postulate revisited." Annual review of neuroscience 24.1
    | (2001): 139-166.

    Args:
        name: the string name of this cell

        shape: tuple specifying shape of this synaptic cable (usually a 2-tuple
            with number of inputs by number of outputs)

        A_plus: strength of long-term potentiation (LTP)

        A_minus: strength of long-term depression (LTD)

        eta: global learning rate initial value/condition (default: 1)

        mu: controls the power scale of the Hebbian shift

        pretrace_target: controls degree of pre-synaptic disconnect, i.e., amount of decay
                 (higher -> lower synaptic values)

        tau_elg: eligibility trace time constant (default: 0); must be >0,
            otherwise, the trace is disabled and this synapse evolves via M-STDP

        elg_decay: eligibility decay constant (default: 1)

        weight_init: a kernel to drive initialization of this synaptic cable's values;
            typically a tuple with 1st element as a string calling the name of
            initialization to use

        resist_scale: a fixed scaling factor to apply to synaptic transform
            (Default: 1.), i.e., yields: out = ((W * Rscale) * in)

        p_conn: probability of a connection existing (default: 1.); setting
            this to < 1. will result in a sparser synaptic structure
    """

    # Define Functions
    def __init__(self, name, shape, A_plus, A_minus, eta=1., mu=0.,
                 pretrace_target=0., tau_elg=0., elg_decay=1.,
                 weight_init=None, resist_scale=1., p_conn=1., w_bound=1.,
                 batch_size=1, **kwargs):
        super().__init__(name, shape, A_plus, A_minus, eta=eta, mu=mu,
                         pretrace_target=pretrace_target, weight_init=weight_init,
                         resist_scale=resist_scale, p_conn=p_conn, w_bound=w_bound,
                         batch_size=batch_size, **kwargs)
        ## MSTDP/MSTDP-ET meta-parameters
        self.tau_elg = tau_elg
        self.elg_decay = elg_decay
        ## MSTDP/MSTDP-ET compartments
        self.modulator = Compartment(jnp.zeros((self.batch_size, 1)))
        self.eligibility = Compartment(jnp.zeros(shape))

    @staticmethod
    def _evolve(dt, w_bound, preTrace_target, mu, Aplus, Aminus, tau_elg,
                elg_decay, preSpike, postSpike, preTrace, postTrace, weights,
                eta, modulator, eligibility):
        ## compute local synaptic update (via STDP)
        dW_dt = TraceSTDPSynapse._compute_update(
            dt, w_bound, preTrace_target, mu, Aplus, Aminus,
            preSpike, postSpike, preTrace, postTrace, weights
        ) ## produce dW/dt (ODE for synaptic change dynamics)
        if tau_elg > 0.: ## perform dynamics of M-STDP-ET
            ## update eligibility trace given current local update
            # dElg_dt = -eligibility * elg_decay + dW_dt * update_scale
            # eligibility = eligibility + dElg_dt * dt/elg_tau
            eligibility = eligibility * jnp.exp(-dt / tau_elg) * elg_decay + dW_dt
        else: ## perform dynamics of M-STDP (no eligibility trace)
            eligibility = dW_dt
        ## Perform a trace/update times a modulatory signal (e.g., reward)
        dWeights = eligibility * modulator

        ## do a gradient ascent update/shift
        weights = weights + dWeights * eta ## modulate update
        ## enforce non-negativity
        eps = 0.01
        weights = jnp.clip(weights, eps, w_bound - eps)  # jnp.abs(w_bound))
        return weights, dWeights, eligibility

    @resolver(_evolve)
    def evolve(self, weights, dWeights, eligibility):
        self.weights.set(weights)
        self.dWeights.set(dWeights)
        self.eligibility.set(eligibility)

    @staticmethod
    def _reset(batch_size, shape):
        preVals = jnp.zeros((batch_size, shape[0]))
        postVals = jnp.zeros((batch_size, shape[1]))
        synVals = jnp.zeros(shape)
        inputs = preVals
        outputs = postVals
        preSpike = preVals
        postSpike = postVals
        preTrace = preVals
        postTrace = postVals
        dWeights = synVals
        eligibility = synVals
        return (inputs, outputs, preSpike, postSpike, preTrace, postTrace,
                dWeights, eligibility)

    @resolver(_reset)
    def reset(self, inputs, outputs, preSpike, postSpike, preTrace, postTrace,
              dWeights, eligibility):
        self.inputs.set(inputs)
        self.outputs.set(outputs)
        self.preSpike.set(preSpike)
        self.postSpike.set(postSpike)
        self.preTrace.set(preTrace)
        self.postTrace.set(postTrace)
        self.dWeights.set(dWeights)
        self.eligibility.set(eligibility)

    @classmethod
    def help(cls): ## component help function
        properties = {
            "synapse_type": "MSTDPETSynapse - performs an adaptable synaptic "
                            "transformation of inputs to produce output signals; "
                            "synapses are adjusted with a form of modulated "
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
                {"weights": "Synapse efficacy/strength parameter values (W)",
                 "eligibility": "Current state of eligibility trace at time `t` (Elg)",
                 "eta": "Global learning rate",
                 "key": "JAX PRNG key"},
            "analytics":
                {"dWeights": "Modulated synaptic weight value adjustment matrix "
                             "produced at time t dW^{stdp}_{ij}/dt"},
            "outputs":
                {"outputs": "Output of synaptic transformation"},
        }
        hyperparams = {
            "shape": "Shape of synaptic weight value matrix; number inputs x number outputs",
            "weight_init": "Initialization conditions for synaptic weight (W) values",
            "batch_size": "Batch size dimension of this component",
            "resist_scale": "Resistance level scaling factor (applied to output of transformation)",
            "p_conn": "Probability of a connection existing (otherwise, it is masked to zero)",
            "A_plus": "Strength of long-term potentiation (LTP)",
            "A_minus": "Strength of long-term depression (LTD)",
            "eta": "Global learning rate initial condition",
            "mu": "Power factor for STDP adjustment",
            "preTrace_target": "Pre-synaptic disconnecting/decay factor (x_tar)",
            "tau_elg": "Eligibility trace time constant",
            "elg_decay": "Eligibility decay factor"
        }
        info = {cls.__name__: properties,
                "compartments": compartment_props,
                "dynamics": "outputs = [(W * Rscale) * inputs] ;"
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
