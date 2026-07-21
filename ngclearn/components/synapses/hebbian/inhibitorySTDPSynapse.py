from jax import random, numpy as jnp
from ngcsimlib.logger import info
from ngcsimlib import deprecate_args
from ngclearn.utils.distribution_generator import DistributionGenerator
from ngclearn import compilable
from ngclearn import Compartment
from ngclearn.components.synapses import DenseSynapse

class InhibitorySTDPSynapse(DenseSynapse): ## inhibitory-STDP synaptic cable
    """
    A synaptic cable that adjusts its efficacies via the trace-based, inhibitory 
    spike-timing-dependent plasticity (iSTDP); this rule can also be configured to 
    utilize a voltage-dependent format of Foldiak anti-Hebbian plasticity. 
.

    | --- Synapse Compartments: ---
    | inputs - input (takes in external signals)
    | outputs - output signals (transformation induced by synapses)
    | weights - current value matrix of synaptic efficacies
    | key - JAX PRNG key
    | --- Synaptic Plasticity Compartments: ---
    | s_pre - pre-synaptic spike to drive 1st term of STDP update (takes in external signals)
    | s_post - post-synaptic spike to drive 2nd term of STDP update (takes in external signals)
    | x_pre - pre-synaptic trace value to drive 1st term of STDP update (takes in external signals)
    | x_post - post-synaptic trace value to drive 2nd term of STDP update (takes in external signals)
    | v_post - post-synaptic voltage/membrane potential state to drive voltage-dependent Foldiak rule (optional)
    | dWeights - current delta matrix containing changes to be applied to synaptic efficacies
    | eta - global learning rate (multiplier beyond A_plus and A_minus)

    | References:
    | Vogels, T.P., Sprekeler, H., Zenke, F., Clopath, C. and Gerstner, W., 2011. Inhibitory 
    | plasticity balances excitation and inhibition in sensory pathways and memory 
    | networks. Science, 334(6062), pp.1569-1573.
    | 
    | Földiak, P., 1990. Forming sparse representations by local anti-Hebbian learning. Biological 
    | cybernetics, 64(2), pp.165-170.

    Args:
        name: the string name of this cell

        shape: tuple specifying shape of this synaptic cable (usually a 2-tuple
            with number of inputs by number of outputs)

        A_plus: strength of long-term potentiation (LTP)

        A_minus: strength of long-term depression (LTD)

        eta: global learning rate initial value/condition (Default: 1)

        rho: controls target firing rate value (Default: 0.2)

        tau_x_pre: pre-synaptic trace time constant  (ms) (Default: 30 ms)

        tau_x_post: post-synaptic trace time constant  (ms) (Default: 30 ms)

        use_soft_bounds: trigggers weight-dependent soft-bounding variant of iSTDP/Foldiak 
            plasticity (Default: False)

        w_max: maximal synaptic efficacy allowed (hard upper bound; Default: 1) 

        w_min: minimal synaptic efficacy allowed (hard lower bound; Default: 0)

        is_voltage_dependent: if True, this synaptic cables adapts via voltage-dependent Foldiak 
            plasticity (Default: False)

        weight_init: a kernel to drive initialization of this synaptic cable's values;
            typically a tuple with 1st element as a string calling the name of
            initialization to use

        bias_init: a kernel to drive initialization to this cable's fixed bias/shift values

        g_conduct_factor: a fixed scaling factor to apply to synaptic transform
            (Default: 1.), i.e., yields: out = ((W * mask) * in) * g_conduct_factor

        p_conn: probability of a connection existing (default: 1); setting
            this to < 1. will result in a sparser synaptic structure

        weight_mask: synaptic binary masking matrix to apply (to enforce a constant sparse structure; default: None)
    """

    @deprecate_args(_rebind=True, w_bound='w_max')
    def __init__(
            self,
            name,
            shape,
            eta=1.,
            rho=0.2, ## target/desired baseline activity level
            Aplus=1., ## weight on LTP term
            Aminus=1., ## weight on LTD term
            tau_x_pre=30.,
            tau_x_post=30.,
            use_soft_bounds=True,
            w_max=1.,
            w_min=0.,
            is_volt_dependent=False, ## triggers voltage-dependent format
            weight_init=None,
            bias_init=None,
            g_conduct_factor=1.,
            p_release_mean=1.0,
            p_conn=1.,
            mask=None, ## None input -> triggers default mask=1
            batch_size=1,
            **kwargs
    ):
        super().__init__(
            name,
            shape=shape,
            weight_init=weight_init,
            bias_init=bias_init,
            g_conduct_factor=g_conduct_factor,
            p_release_mean=p_release_mean,
            p_conn=p_conn,
            batch_size=batch_size,
            mask=mask,
            **kwargs
        )
        ## key anti-Hebbian (iSTDP) synapse meta-parameters
        self.eta = eta
        self.Aplus = Aplus # 5-10 -> scales up rare coincidences in LTP
        self.Aminus = Aminus
        self.use_soft_bounds = use_soft_bounds
        self.tau_x_pre = tau_x_pre
        self.tau_x_post = tau_x_post
        self.rho = rho
        self.w_max = w_max
        self.w_min = w_min
        self.is_volt_dependent = is_volt_dependent

        ## set up anti-Hebbian synapse key (extra) compartments
        _pre_reset = jnp.zeros((batch_size, shape[0]))
        _post_reset = jnp.zeros((batch_size, shape[1]))
        self.s_pre = Compartment(_pre_reset) ## input compartment
        self.x_pre = Compartment(_pre_reset) ## internal
        self.s_post = Compartment(_post_reset) ## input compartment
        self.x_post = Compartment(_post_reset) ## internal
        self.v_post = Compartment(_post_reset) ## input compartment (optional)

        self.dWeights = Compartment(jnp.zeros(shape))
        self.weights.set(self.weights.get() * self.mask.get()) ## make sure mask is enforced

    # @compilable
    # def advance_state(self, t, dt): ## masked synaptic transmission
    #     W = self.weights.get()
    #     W = W * self.mask.get()
    #     inputs = self.inputs.get() ## reuses incoming cable input signals
    #     self.outputs.set(
    #         (jnp.matmul(inputs, W) * self.g_conduct_factor) + self.biases.get()
    #     )

    @compilable
    def evolve(self, t, dt): ## NOTE: spike-based anti-Hebbian rule
        W = self.weights.get()
        s_pre = self.s_pre.get() ## pre-synaptic inhibitory spikes
        s_post = self.s_post.get()
        x_pre = self.x_pre.get() ## filtered inhibitory spikes
        x_post = self.x_post.get() ## post-synaptic target's spikes
        v_post = self.v_post.get() ## post-synaptic target's voltage

        ## (low-pass) filter synaptic spikes over time
        x_pre = x_pre + (-x_pre + s_pre) * dt/self.tau_x_pre ## pre-trace
        self.x_pre.set(x_pre)
        x_post = x_post + (-x_post + s_post) * dt/self.tau_x_post ## post-trace
        self.x_post.set(x_post)

        batch_size = x_pre.shape[0]
        if self.is_volt_dependent: ## trigger voltage-dependent foldiak anti-Hebbian rule
            ## this simple NAC lab rule assumes: 
            ### W - excitatory-to-inhibitory synaptic efficacies
            ### x_pre - inhibitory spike trace (shape = batch_size x num_inh)
            ### v_post - excitatory voltage (shape = batch_size x num_exc)
            ### rho - target voltage rate
            ## compute (scaled) synaptic adjustment
            dW = (x_pre.T @ (v_post - self.rho)) / batch_size
            bound_scale = jnp.where(dW > 0, self.w_max - W, W)
            dW = dW * bound_scale
        else:
            ## Vogels-Sprekeler rule iSTDP assumes:
            ### W - excitatory-to-inhibitory synaptic efficacies
            ### x_pre - inhibitory spike trace (shape = batch_size x num_inh)
            ### s_pre - raw inhibitory spikes  (shape = batch_size x num_inh)
            ### x_post - excitatory spike trace (shape = batch_size x num_exc)
            ### s_post - raw excitatory spikes  (shape = batch_size x num_exc)
            ### rho - target firing rate fraction

            ## calculate Vogels-Sprekeler coincidence + homeostatic matrices
            potentiation = ( (x_pre.T @ s_post) + (s_pre.T @ x_post) ) ## coincidence matrix shape = num_inh x num_exc
            ## target-rate suppression matrix (broadcasted across excitatory dimension)
            ### this depends on x_pre (scales depression based on recent inhibitory activity)
            ### 2 * rho * x_pre(t)  (where ltd_bias = 2)
            ltd_bias = 2.0 ## NOTE: ltd bias value could be increased if needed
            depression = (x_pre.T @ jnp.ones_like(s_post)) * (ltd_bias * self.rho)
            ## compute raw weight change averaged over batch-length
            ltp = potentiation
            ltd = -depression
            if self.use_soft_bounds: ## apply weight-dependency (NAC-lab extension)
                ltp = potentiation * (self.w_max - W)
                ltd = -depression * W
            dW = (ltp * self.Aplus + ltd * self.Aminus) / batch_size

        ## apply update rule to adjust synaptic efficacies
        W = W + dW * self.eta
        W = jnp.clip(W, self.w_min, self.w_max) ## constrains W to stay w/in bounds
        self.weights.set(W * self.mask.get())
        self.dWeights.set(dW)

    @compilable
    def reset(self):
        in_reset_vals = jnp.zeros((self.batch_size, self.shape[0]))
        out_reset_vals = jnp.zeros((self.batch_size, self.shape[1]))
        if not self.inputs.targeted:
            self.inputs.set(in_reset_vals)
        self.x_pre.set(in_reset_vals)
        self.s_pre.set(in_reset_vals)
        self.outputs.set(out_reset_vals)
        self.s_post.set(out_reset_vals)
        self.x_post.set(out_reset_vals)
        if not self.v_post.targeted:
            self.v_post.set(out_reset_vals)
        self.dWeights.set(self.dWeights.get() * 0)

    @classmethod
    def help(cls): ## component help function
        properties = {
            "synapse_type": "InhibitorySTDPSynapse - performs an adaptable synaptic "
                            "transformation of inputs to produce output signals; "
                            "synapses are adjusted with trace-based "
                            "inhibitory spike-timing-dependent plasticity (iSTDP)"
        }
        compartment_props = {
            "inputs":
                {"inputs": "Takes in external input signal values",
                 "s_pre": "Pre-synaptic spike compartment value/term for iSTDP (s_j)",
                 "s_post": "Post-synaptic spike compartment value/term for iSTDP (s_i)",
                 "v_post": "Post-synaptic voltage value term for Foldiak anti-Hebbian format (v_i)"},
            "states":
                {"weights": "Synapse efficacy/strength parameter values",
                 "biases": "Base-rate/bias parameter values",
                 "eta": "Global learning rate (multiplier beyond A_plus and A_minus)",
                 "x_pre": "Pre-synaptic trace value term for iSTDP (z_j)",
                 "x_post": "Post-synaptic trace value term for iSTDP (z_i)", 
                 "key": "JAX PRNG key"},
            "analytics":
                {"dWeights": "Synaptic weight value adjustment matrix produced at time t"},
            "outputs":
                {"outputs": "Output of synaptic transformation"},
        }
        hyperparams = {
            "shape": "Shape of synaptic weight value matrix; number inputs x number outputs",
            "batch_size": "Batch size dimension of this component",
            "weight_init": "Initialization conditions for synaptic weight (W) values",
            "g_conduct_factor": "Conductance level scaling factor (applied to output of transformation)",
            "p_conn": "Probability of a connection existing (otherwise, it is masked to zero)",
            "A_plus": "Strength of long-term potentiation (LTP)",
            "A_minus": "Strength of long-term depression (LTD)",
            "eta": "Global learning rate initial condition",
            "weight_mask" : "Binary synaptic weight mask to apply to enforce a sparsity structure"
        }
        info = {cls.__name__: properties,
                "compartments": compartment_props,
                "dynamics": "outputs = [(W * g_conduct_factor) * inputs] ;"
                            "dW_{ij}/dt = A_plus * (z_j * s_i + s_j * z_i) - A_minus * (2 * rho * z_j)",
                "hyperparameters": hyperparams}
        return info 

