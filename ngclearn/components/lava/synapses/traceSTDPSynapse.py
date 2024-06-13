from ngclearn import resolver, Component, Compartment
from ngclearn.utils import tensorstats
from ngclearn import numpy as jnp
from ngclearn.utils.weight_distribution import initialize_params
from ngcsimlib.logger import info, warn

class TraceSTDPSynapse(Component): ## Lava-compliant trace-STDP synapse
    """
    A synaptic cable that adjusts its efficacies via trace-based form of
    spike-timing-dependent plasticity (STDP). This is a Lava-compliant synaptic
    cable that adjusts with a hard-coded form of (stochastic) gradient ascent.

    | --- Synapse Input Compartments: (Takes wired-in signals) ---
    | inputs - input (pre-synaptic) stimulus
    | --- Synaptic Plasticity Input Compartments: (Takes in wired-in signals) ---
    | pre - pre-synaptic spike(s) to drive STDP update
    | x_pre - pre-synaptic trace value(s) to drive STDP update
    | post - post-synaptic spike(s) to drive STDP update
    | x_post - post-synaptic trace value(s) to drive STDP update
    | eta - global learning rate (unidimensional/scalar value)
    | --- Synapse Output Compartments: (These signals are generated) ---
    | outputs - transformed (post-synaptic) signal
    | weights - current value matrix of synaptic efficacies (this is post-update if eta > 0)

    Args:
        name: the string name of this cell

        dt: integration time constant (ms)

        resist_scale: a fixed scaling factor to apply to synaptic transform
            (Default: 1.), i.e., yields: out = ((W * Rscale) * in) + b

        weight_init: a kernel to drive initialization of this synaptic cable's values;
            typically a tuple with 1st element as a string calling the name of
            initialization to use

        shape: tuple specifying shape of this synaptic cable (usually a 2-tuple
            with number of inputs by number of outputs)

        Aplus: strength of long-term potentiation (LTP)

        Aminus: strength of long-term depression (LTD)

        eta: global learning rate (default: 1)

        w_decay: degree to which (L2) synaptic weight decay is applied to the
            computed Hebbian adjustment (Default: 0); note that decay is not
            applied to any configured biases

        w_bound: maximum weight to softly bound this cable's value matrix to; if
            set to 0, then no synaptic value bounding will be applied

        preTrace_target: controls degree of pre-synaptic disconnect, i.e., amount of decay
                 (higher -> lower synaptic values)

        weights: matrix of synaptic weight values to initialize this synapse
            component to

        Rscale: DEPRECATED argument (maps to resist_scale)
    """

    # Define Functions
    def __init__(self, name, dt, resist_scale=1., weight_init=None, shape=None,
                 Aplus=0.01, Aminus=0.001, eta=1., w_decay=0., w_bound=1.,
                 preTrace_target=0., weights=None, **kwargs):
        super().__init__(name, **kwargs)

        ## synaptic plasticity properties and characteristics
        self.weight_init = weight_init
        self.shape = shape
        self.dt = dt
        self.Rscale = resist_scale
        if kwargs.get("Rscale") is not None:
            warn("The argument `Rscale` being used is deprecated.")
            self.Rscale = kwargs.get("Rscale")
        self.w_bounds = w_bound
        self.w_decay = w_decay ## synaptic decay
        self.eta0 = eta
        self.Aplus = Aplus
        self.Aminus = Aminus
        self.x_tar = preTrace_target

        ## Component size setup
        self.batch_size = 1

        self.eta = Compartment(jnp.ones((1, 1)) * eta)

        self.inputs = Compartment(None)
        self.outputs = Compartment(None)
        self.pre = Compartment(None) ## pre-synaptic spike
        self.x_pre = Compartment(None) ## pre-synaptic trace
        self.post = Compartment(None) ## post-synaptic spike
        self.x_post = Compartment(None) ## post-synaptic trace
        self.weights = Compartment(None)

        if weights is not None:
            warn("The argument `weights` being used is deprecated.")
            self._init(weights)
        else:
            assert self.shape is not None ## if using an init, MUST have shape
            if self.weight_init is None:
                info(self.name, "is using default weight initializer!")
                self.weight_init = {"dist": "uniform", "amin": 0.025,
                                    "amax": 0.8}
            weights = initialize_params(None, self.weight_init, self.shape)
            self._init(weights)

    def _init(self, weights):
        self.rows = weights.shape[0]
        self.cols = weights.shape[1]
        ## pre-computed empty zero pads
        preVals = jnp.zeros((self.batch_size, self.rows))
        postVals = jnp.zeros((self.batch_size, self.cols))
        ## Compartments
        self.inputs.set(preVals)
        self.outputs.set(postVals)
        self.pre.set(preVals) ## pre-synaptic spike
        self.x_pre.set(preVals) ## pre-synaptic trace
        self.post.set(postVals) ## post-synaptic spike
        self.x_post.set(postVals) ## post-synaptic trace
        self.weights.set(weights)

    @staticmethod
    def _advance_state(dt, Rscale, Aplus, Aminus, w_bounds, w_decay, x_tar,
                       inputs, weights, pre, x_pre, post, x_post, eta):
        outputs = jnp.matmul(inputs, weights) * Rscale
        ########################################################################
        ## Run one step of STDP online
        dWpost = jnp.matmul((x_pre - x_tar).T, post * Aplus)
        dWpre = -jnp.matmul(pre.T, x_post * Aminus)
        dW = dWpost + dWpre
        ## reformulated bounding flag to be linear algebraic
        flag = (w_bounds > 0.) * 1.
        dW = (dW * (w_bounds - jnp.abs(weights))) * flag + (dW) * (1. - flag)
        ## physically adjust synapses
        weights = weights + (dW - weights * w_decay) * eta
        #weights = weights + (dW - weights * w_decay) * dt/tau_w ## ODE format
        weights = jnp.clip(weights, 0., w_bounds)
        ########################################################################
        return outputs, weights

    @resolver(_advance_state)
    def advance_state(self, outputs, weights):
        self.outputs.set(outputs)
        self.weights.set(weights)

    @staticmethod
    def _reset(batch_size, rows, cols, eta0):
        preVals = jnp.zeros((batch_size, rows))
        postVals = jnp.zeros((batch_size, cols))
        return (
            preVals, # inputs
            postVals, # outputs
            preVals, # pre
            postVals, # post
            preVals, # x_pre
            postVals, # x_post
            jnp.ones((1, 1)) * eta0
        )

    @resolver(_reset)
    def reset(self, inputs, outputs, pre, post, x_pre, x_post, eta):
        self.inputs.set(inputs)
        self.outputs.set(outputs)
        self.pre.set(pre)
        self.post.set(post)
        self.x_pre.set(x_pre)
        self.x_post.set(x_post)
        self.eta.set(eta)

    def save(self, directory, **kwargs):
        file_name = directory + "/" + self.name + ".npz"
        jnp.savez(file_name, weights=self.weights.value)

    def load(self, directory, **kwargs):
        file_name = directory + "/" + self.name + ".npz"
        data = jnp.load(file_name)
        self._init( data['weights'] )

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
