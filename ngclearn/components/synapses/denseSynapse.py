from jax import random, numpy as jnp, jit
from ngclearn.components.jaxComponent import JaxComponent
from ngclearn.utils.distribution_generator import DistributionGenerator
from ngcsimlib.logger import info
from ngcsimlib import deprecate_args

from ngclearn import compilable #from ngcsimlib.parser import compilable
from ngclearn import Compartment #from ngcsimlib.compartment import Compartment

class DenseSynapse(JaxComponent): ## base dense synaptic cable
    """
    A dense synaptic cable; no form of synaptic evolution/adaptation
    is in-built to this component.

    | --- Synapse Input Compartments: ---
    | inputs - input (takes in external signals)
    | --- Synapse State Compartments: ---
    | weights - current value matrix of synaptic efficacies (strength values)
    | biases - current value vector of synaptic bias values
    | gate - current values of multiplicative (output) gate/modulator (Default: 1)
    | --- Synapse Output Compartments: ---
    | outputs - output signals

    Args:
        name: the string name of this cell

        shape: tuple specifying shape of this synaptic cable (usually a 2-tuple
            with number of inputs by number of outputs)

        weight_init: a kernel to drive initialization of this synaptic cable's values;
            typically a tuple with 1st element as a string calling the name of
            initialization to use

        bias_init: a kernel to drive initialization of biases for this synaptic cable
            (Default: None, which turns off/disables biases)

        g_conduct_factor: a fixed (resistance) scaling factor to apply to synaptic
            transform (Default: 1.), i.e., yields: out = ((W * in) * g_conduct_factor) + bias

        p_release_mean: probability of pre-synaptic transmission; only if this value is > 0 and < 1, 
            this synapse will enforce stochastic synaptic tranmission on pre-synaptic signals, 
            meaning that each pre-synaptic signal will make it across the synaptic cable with 
            a probability of `p(transmit) = p_release_mean +/- 0.1` (Default: 1) 

        p_conn: probability of a connection existing (default: 1.); setting
            this to < 1 and > 0. will result in a sparser synaptic structure
            (lower values yield sparse structure)

        max_delay_steps: maximum delay length (in terms of discrete simulation time-steps) to 
            delay transmission of pre-synaptic signals across this synaptic cable; note 
            that setting this to 0 disables the use of synaptic delay (Default: 0)

        mask: if non-None, a (multiplicative) mask is applied to this synaptic weight matrix
    """

    @deprecate_args(_rebind=True, resist_scale='g_conduct_factor')
    def __init__(
            self,
            name,
            shape,
            weight_init=None,
            bias_init=None,
            g_conduct_factor=1.,
            p_release_mean=1.,
            p_conn=1.,
            max_delay_steps=0, ## "Tk"
            mask=None,
            batch_size=1,
            **kwargs
    ):
        super().__init__(name, **kwargs)

        self.batch_size = batch_size
        # self.mask = 1.
        # if mask is not None:
        #     self.mask = mask

        ## Synapse meta-parameters
        self.shape = shape
        self.g_conduct_factor = g_conduct_factor
        self.p_release_mean = p_release_mean

        ## Set up synaptic weight values
        tmp_key, *subkeys = random.split(self.key.get(), 4)

        if weight_init is None:
            info(self.name, "is using default weight initializer!")
            weight_init = DistributionGenerator.uniform(0.025, 0.8)
        weights = weight_init(shape, subkeys[0])

        if 0. < p_conn < 1.: ## modifier/constraint: only non-zero and <1 probs allowed
            p_mask = random.bernoulli(subkeys[1], p=p_conn, shape=shape)
            weights = weights * p_mask ## sparsify matrix

        ## Compartment setup
        preVals = jnp.zeros((self.batch_size, shape[0]))
        postVals = jnp.zeros((self.batch_size, shape[1]))

        self.inputs = Compartment(preVals)
        self.outputs = Compartment(postVals)
        self.weights = Compartment(weights)
        _mask = jnp.ones((1, 1))
        if mask is not None:
            _mask = mask
        self.mask = Compartment(_mask)
        ## Set up (optional) bias values
        if bias_init is None:
            info(self.name, "is using default bias value of zero (no bias kernel provided)!")
        self.biases = Compartment(bias_init((1, shape[1]), subkeys[2]) if bias_init else 0.0)
        self.gate = Compartment(postVals + 1.)
        ## pin weight/bias initializers to component
        self.weight_init = weight_init
        self.bias_init = bias_init

        ## Stochastic synaptic transmission - create static vector of heterogeneous release probabilities
        key, *skey = random.split(self.key.get(), 4)
        pre_units = shape[0]
        self.p_release_mean = p_release_mean
        p_jitter = 0.1 ## NOTE: this is hard-coded jitter
        self.use_one_spike = True #False
        self.p_release = jnp.ones((1, pre_units))
        if 0. < self.p_release_mean < 1.: ## if proper p(transmit) mean given
            self.p_release = random.uniform(
                skey[0], shape=(1, pre_units), minval=self.p_release_mean - p_jitter, maxval=p_release_mean + p_jitter
            )  ## probability of spike release

        ## Implement staggered (pre-synaptic/axonal) cable delays
        self.max_delay_steps = max_delay_steps ## extends up to a `Tk` ms temporal jitter window
        pre_units = shape[0] 
        ## jitter axonal delays from 0 up to Tk
        self.syn_delay_indices = random.randint(skey[1], shape=(pre_units,), minval=0, maxval=self.max_delay_steps) 
        ## create fixed memory grid to store a rolling history of incoming pre-synaptic signals
        initial_buffer_state = jnp.zeros((self.max_delay_steps, self.batch_size, pre_units))
        self.input_delay_buffer = Compartment(initial_buffer_state, display_name="Synaptic Input Queue")
        self.delayed_inputs = Compartment(preVals) ## record of actual emitted delayed inputs (if delay > 0)

    @compilable
    def advance_state(self):
        gate = self.gate.get()
        weights = self.weights.get()
        weights = weights * self.mask.get()
        raw_inputs = self.inputs.get()

        inputs = raw_inputs
        if self.max_delay_steps > 0: ## implements synaptic jitter via axonal delay
            buffer = self.input_delay_buffer.get()
            ## gather historical timestep slice, independently for each input axon line
            time_indices = self.syn_delay_indices  ## shape: (D_pre,)
            pre_indices = jnp.arange(raw_inputs.shape[1])  ## shape: (D_pre,)
            ## advanced gather outputs parallelized, jittered spike matrix (shape: (batch_size, D_pre))
            inputs = buffer[time_indices, :, pre_indices].T ## get delay pre-synaptic signal(s)
            ## roll input conveyor belt forward - shift historical slots down (& drop oldest timestep)
            rolled_buffer = jnp.roll(buffer, shift=-1, axis=0)
            ## update buffer - current input spikes 'raw_inputs' go to back of queue
            updated_buffer = rolled_buffer.at[-1, :, :].set(raw_inputs)
            self.input_delay_buffer.set(updated_buffer)
        ## else, leave inputs = raw_inputs untouched

        if 0. < self.p_release_mean < 1.: ## engage in stochastic synaptic transmission (in probability form)
            ## Reference: 
            ## Del Castillo, J. and Katz, B., 1954. Quantal components of the end-plate potential.
            ## The Journal of physiology, 124(3), p.560.
            p_matrix = self.p_release ## get per-neuron release probs
            key, skey = random.split(self.key.get(), 2) ## generate random Bernoulli mask
            if not self.use_one_spike: ## does per-neuron sampled firing
                release_mask = (random.uniform(skey, shape=raw_inputs.shape) < p_matrix).astype(jnp.float32)
            else: ## does blind "guarantee-one-signal-fires" sampling
                rP = raw_inputs * random.uniform(skey, raw_inputs.shape)
                release_mask = nn.one_hot(jnp.argmax(rP, axis=1), num_classes=raw_inputs.shape[1], dtype=jnp.float32)
            ## apply stochastic transmission: fully sparse, event-driven signals
            inputs = raw_inputs * release_mask
            self.key.set(key) ## update noise key of this component
        ## else, leave inputs un-corrupted/untouched

        self.delayed_inputs.set(inputs) ## store emitted delayed inputs
        ## carry signals across synaptic cable
        self.outputs.set((jnp.matmul(inputs, weights) * gate * self.g_conduct_factor) + self.biases.get())

    @compilable
    def reset(self):
        if not self.inputs.targeted:
            self.inputs.set(jnp.zeros((self.batch_size, self.shape[0])))
        if not self.gate.targeted:
            self.gate.set(jnp.ones((self.batch_size, self.shape[1])))
        self.delayed_inputs.set(self.delayed_inputs.get() * 0)
        self.outputs.set(jnp.zeros((self.batch_size, self.shape[1])))
        self.input_delay_buffer.set(self.input_delay_buffer.get() * 0)

    @classmethod
    def help(cls): ## component help function
        properties = {
            "synapse_type": "DenseSynapse - performs a synaptic transformation "
                            "of inputs to produce  output signals (e.g., a "
                            "scaled linear multivariate transformation)"
        }
        compartment_props = {
            "inputs":
                {"inputs": "Takes in external input signal values"},
            "states":
                {"weights": "Synapse efficacy/strength parameter values",
                 "biases": "Base-rate/bias parameter values",
                 "key": "JAX PRNG key"},
            "outputs":
                {"outputs": "Output of synaptic transformation"},
        }
        hyperparams = {
            "shape": "Shape of synaptic weight value matrix; number inputs x number outputs",
            "batch_size": "Batch size dimension of this component",
            "weight_init": "Initialization conditions for synaptic weight (W) values",
            "bias_init": "Initialization conditions for bias/base-rate (b) values",
            "g_conduct_factor": "Conductance/average level scaling factor; applied to output of transformation",
            "p_conn": "Probability of a connection existing (otherwise, it is masked to zero)", 
            "p_release_mean": "Probability of pre-synaptic signal firing across axon and into synaptic cable line", 
            "max_delay_steps": "Maximum number of simulation steps to delay signal making it over axon & into cable line"
        }
        info = {cls.__name__: properties,
                "compartments": compartment_props,
                "dynamics": "outputs = [W * inputs] * Rscale + b",
                "hyperparameters": hyperparams}
        return info

if __name__ == '__main__':
    from ngcsimlib.context import Context
    with Context("Bar") as bar:
        Wab = DenseSynapse("Wab", (2, 3))
    print(Wab)
