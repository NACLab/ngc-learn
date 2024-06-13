from ngclearn import numpy as jnp
from ngclearn import resolver, Component, Compartment
from ngclearn.utils import tensorstats
from ngclearn.utils.weight_distribution import initialize_params
from ngcsimlib.logger import info, warn

class StaticSynapse(Component): ## Lava-compliant fixed/non-evolvable synapse
    """
    A static (dense) synaptic cable; no form of synaptic evolution/adaptation
    is in-built to this component. This a Lava-compliant version of the
    static synapse component from the synapses sub-package of components.

    | --- Synapse Input Compartments: (Takes wired-in signals) ---
    | inputs - input (pre-synaptic) stimulus
    | --- Synapse Output Compartments: .set()ese signals are generated) ---
    | outputs - transformed (post-synaptic) signal
    | weights - current value matrix of synaptic efficacies (this is post-update if eta > 0)

    Args:
        name: the string name of this cell

        dt: integration time constant (ms)

        weight_init: a kernel to drive initialization of this synaptic cable's values;
            typically a tuple with 1st element as a string calling the name of
            initialization to use

        shape: tuple specifying shape of this synaptic cable (usually a 2-tuple
            with number of inputs by number of outputs)

        resist_scale: a fixed scaling factor to apply to synaptic transform
            (Default: 1.), i.e., yields: out = ((W * Rscale) * in) + b

        Rscale: DEPRECATED argument (maps to resist_scale)

        weights: a provided, externally created weight value matrix that will
            be used instead of an auto-init call
    """

    # Define Functions
    def __init__(self, name, dt, weight_init=None, shape=None, resist_scale=1.,
                 weights=None, **kwargs):
        super().__init__(name, **kwargs)

        ## synaptic plasticity properties and characteristics
        self.batch_size = 1
        self.dt = dt
        self.Rscale = resist_scale
        if kwargs.get("Rscale") is not None:
            warn("The argument `Rscale` being used is deprecated.")
            self.Rscale = kwargs.get("Rscale")
        self.shape = shape
        self.weight_init = weight_init

        self.inputs = Compartment(None)
        self.outputs = Compartment(None)
        self.weights = Compartment(None)

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
        self.weights.set(weights)

    @staticmethod
    def _advance_state(dt, Rscale, inputs, weights):
        outputs = jnp.matmul(inputs, weights) * Rscale
        return outputs

    @resolver(_advance_state)
    def advance_state(self, outputs):
        self.outputs.set(outputs)

    @staticmethod
    def _reset(batch_size, rows, cols):
        preVals = jnp.zeros((batch_size, rows))
        postVals = jnp.zeros((batch_size, cols))
        return (
            preVals, # inputs
            postVals, # outputs
        )

    @resolver(_reset)
    def reset(self, inputs, outputs):
        self.inputs.set(inputs)
        self.outputs.set(outputs)

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
