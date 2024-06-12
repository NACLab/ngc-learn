from ngclearn import resolver, Component, Compartment
from ngclearn.utils import tensorstats
from ngclearn import numpy as jnp
from ngclearn.utils.weight_distribution import initialize_params
from ngcsimlib.logger import info, warn

class HebbianSynapse(Component): ## Lava-compliant Hebbian synapse
    """
    A synaptic cable that adjusts its efficacies via a two-factor Hebbian
    adjustment rule. This is a Lava-compliant synaptic cable that adjusts
    with a hard-coded form of (stochastic) gradient ascent.

    | --- Synapse Input Compartments: (Takes wired-in signals) ---
    | inputs - input (pre-synaptic) stimulus
    | --- Synaptic Plasticity Input Compartments: (Takes in wired-in signals) ---
    | pre - pre-synaptic signal to drive first term of Hebbian update
    | post - post-synaptic signal to drive 2nd term of Hebbian update
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

        eta: global learning rate

        w_decay: degree to which (L2) synaptic weight decay is applied to the
            computed Hebbian adjustment (Default: 0); note that decay is not
            applied to any configured biases

        w_bound: maximum weight to softly bound this cable's value matrix to; if
            set to 0, then no synaptic value bounding will be applied

        weights: matrix of synaptic weight values to initialize this synapse
            component to

        Rscale: DEPRECATED argument (maps to resist_scale)
    """

    # Define Functions
    def __init__(self, name, dt, resist_scale=1., weight_init=None, shape=None,
                 eta=0., w_decay=0., w_bound=1., weights=None, **kwargs):
        super().__init__(name, **kwargs)

        ## synaptic plasticity properties and characteristics
        self.weight_init = weight_init
        self.shape = shape
        self.batch_size = 1

        self.dt = dt
        self.Rscale = resist_scale
        if kwargs.get("Rscale") is not None:
            warn("The argument `Rscale` being used is deprecated.")
            self.Rscale = kwargs.get("Rscale")
        self.w_bounds = w_bound
        self.w_decay = w_decay ## synaptic decay
        self.eta0 = eta

        self.inputs = Compartment(None)
        self.outputs = Compartment(None)
        self.pre = Compartment(None)
        self.post = Compartment(None)
        self.weights = Compartment(None)
        self.eta = Compartment(jnp.ones((1, 1)) * eta)

        if weights is not None:
            warn("The argument `weights` being used is deprecated.")
            self._init(weights)
        else:
            assert self.shape is not None  ## if using an init, MUST have shape
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
        self.pre.set(preVals)
        self.post.set(postVals)
        self.weights.set(weights)

    @staticmethod
    def _advance_state(dt, Rscale, w_bounds, w_decay, inputs, weights,
                       pre, post, eta):
        outputs = jnp.matmul(inputs, weights) * Rscale
        ########################################################################
        ## Run one step of 2-factor Hebbian adaptation online
        dW = jnp.matmul(pre.T, post)
        #db = jnp.sum(_post, axis=0, keepdims=True)
        ## reformulated bounding flag to be linear algebraic
        flag = (w_bounds > 0.) * 1.
        dW = (dW * (w_bounds - jnp.abs(weights))) * flag + (dW) * (1. - flag)
        ## add small amount of synaptic decay
        weights = weights + (dW - weights * w_decay) * eta
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
            jnp.ones((1,1)) * eta0
        )

    @resolver(_reset)
    def reset(self, inputs, outputs, pre, post, eta):
        self.inputs.set(inputs)
        self.outputs.set(outputs)
        self.pre.set(pre)
        self.post.set(post)
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
