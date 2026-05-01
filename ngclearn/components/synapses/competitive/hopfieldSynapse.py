from jax import random, numpy as jnp, jit, vmap
from ngclearn import compilable #from ngcsimlib.parser import compilable
from ngclearn import Compartment #from ngcsimlib.compartment import Compartment
from ngclearn.utils.model_utils import softmax, bkwta

from ngclearn.components.synapses.denseSynapse import DenseSynapse

class HopfieldSynapse(DenseSynapse): # (Modern) Hopfield synaptic cable
    """
    A synaptic cable that emulates a modern Hopfield network (MHN). Note that this model has been generalized a bit,
    a.l.a. NAC-Lab style, and comes equipped with two non-standard local plasticity update rules to alter the
    underlying memory matrix W from scratch (or to fine-tune an existing preloaded one); note that a mixed
    MHN can be created (one where initial patterns are stored but portions / elements of the
    memory matrix are further adapted in accordance to a local adjustment rule). This model currently only implements
    the exponential coupling/energy function.

    | --- Synapse Compartments: ---
    | inputs - input probe (takes in external signals)
    | outputs - output signals (retrieved memory / updated probe)
    | weights - current value matrix of synaptic efficacies
    | similarities - current raw similarity scores computed (pre-softmax)
    | memory_weights - current similarity scores computed (post-softmax)
    | i_tick - current internal tick / marker (gets incremented by 1 for each call to `evolve`)
    | energy - current energy functional reading (given current clamped input probe)
    | key - JAX PRNG key
    | --- Synaptic Plasticity Compartments: ---
    | dWeights - current delta matrix containing changes to be applied to synapses

    | References:
    | Movellan, Javier R. "Contrastive Hebbian learning in the continuous Hopfield model." Connectionist models.
    | Morgan Kaufmann, 1991. 10-17.
    |
    | Krotov, Dmitry, and John Hopfield. "Large associative memory problem in neurobiology and machine learning."
    | arXiv preprint arXiv:2008.06996 (2020).
    |
    | Hintzman, Douglas L. "MINERVA 2: A simulation model of human memory." Behavior Research Methods, Instruments,
    | & Computers 16.2 (1984): 96-101.

    Args:
        name: the string name of this cell

        shape: tuple specifying shape of this synaptic cable (usually a 2-tuple
            with number of inputs by number of outputs)

        eta: (initial) learning rate / step-size for this SOM (initial condition value for `eta`)

        reg_lambda: weight decay coefficient applied to Hebbian update

        beta: (inverse) temperature to control sharpness of memory similarity calculation

        initial_patterns: seed patterns to store within memory matrix (Default: None)

        update_rule: local plasticity rule to use to adjust/update memory matrix (Default: "delta";
            Currently, two rules are encoded that work - a custom delta rule (prescribed error rule) and
            a custom contrastive Hebbian rule (Movellan/NAC-Lab-style)

        weight_init: a kernel to drive initialization of this synaptic cable's values;
            typically a tuple with 1st element as a string calling the name of
            initialization to use

        resist_scale: a fixed scaling factor to apply to synaptic transform
            (Default: 1.), i.e., yields: out = ((W * Rscale) * in)

        p_conn: probability of a connection existing (default: 1.); setting
            this to < 1. will result in a sparser synaptic structure
    """

    def __init__(
            self,
            name,
            shape,
            eta,
            reg_lambda=0.,
            beta=8.,
            initial_patterns=None,
            update_rule = "delta", ## memory plasticity rule
            weight_init=None,
            resist_scale=1.,
            p_conn=1.,
            batch_size=1,
            **kwargs
    ):
        super().__init__(
            name, shape, weight_init, None, resist_scale, p_conn, batch_size=batch_size, **kwargs
        )

        ### Synapse and Hopfield hyper-parameters
        self.eta = eta
        self.reg_lambda = reg_lambda #0.0001 ## regularization co-efficient
        self.l1_lambda = 0. #0.0001 ## coefficient for L1 decay
        self.beta = beta
        if initial_patterns is not None: ## preload memory synaptic matrix
            W = self.weights.get()
            D, H = W.shape
            tmp_key, *subkeys = random.split(self.key.get(), 3)
            if initial_patterns.shape[1] < H: ## randomly portions of memory with stored patterns/templates
                ptrs = random.permutation(subkeys[0], H)
                W = jnp.concat([initial_patterns, W[:, 0:(H - initial_patterns.shape[1])]], axis=1)
                W = W[:, ptrs] ## shuffle memories
                self.weights.set(W)
            else: ## memory is exactly the set of stored patterns/templates
                self.weights.set(initial_patterns)
        self.rule_fx = 2 ## Default: delta-rule
        if update_rule == "contrastive":
            self.rule_fx = 1

        ## Hopfield Compartment setup
        inputVals = jnp.zeros((self.batch_size, shape[0]))
        simVals = jnp.zeros((self.batch_size, shape[0]))
        self.inputs = Compartment(inputVals) ## input shape = output shape
        self.outputs = Compartment(inputVals) ## output shape = input shape
        self.similarities = Compartment(simVals) ## "hidden layer"
        self.memory_weights = Compartment(simVals)

        self.energy = Compartment(jnp.zeros((1, 1)), display_name="Energy")
        self.i_tick = Compartment(jnp.zeros((1, 1)))
        self.dWeights = Compartment(self.weights.get() * 0)

    @compilable
    def advance_state(self): ## forward-inference step of SOM
        WX = self.weights.get()
        probe_t = self.inputs.get()

        ## TODO: what about power/quadratic functions instead? (integrate Minerva power coupling)
        sims = jnp.matmul(probe_t, WX)  ## similarities (w/ xn as probe)
        sims_max = jnp.max(sims, axis=1, keepdims=True)
        sims = sims - sims_max
        self.similarities.set(sims) ## similarities = "hidden layer"
        memory_weights = softmax(sims * self.beta)
        self.memory_weights.set(memory_weights)
        z = memory_weights
        probe_tp1 = jnp.matmul(z, WX.T)  ## calc probe update
        self.outputs.set(probe_tp1)

        ## Calculate (modern) Hopfield energy functional
        N = WX.shape[1]  ## how many neural memories are there
        max_sim_value = jnp.max(self.beta * sims, axis=1, keepdims=True)
        lse = max_sim_value + jnp.log(jnp.sum(jnp.exp(self.beta * sims - max_sim_value), axis=1, keepdims=True))
        term1 = -(1. / self.beta) * lse
        term2 = 0.5 * jnp.expand_dims(jnp.diag(jnp.matmul(probe_t, probe_tp1.T)),axis=1)
        term3 = (1. / self.beta) * jnp.log(N) + 0.5 * jnp.max(jnp.linalg.norm(WX, ord=2, axis=1) ** 2)  ## C
        Ex = jnp.mean(term1 + term2 + term3, axis=0, keepdims=True) #* (1. / probe_t.shape[0]) ## calc batch avg energy
        self.energy.set(Ex)

        self.i_tick.set(self.i_tick.get() + 1.) ## march internal tick forward

    @compilable
    def evolve(self, t, dt):  ## plasticity rule for changing this Hopfield network's memory matrix
        x = self.inputs.get()
        x_hat = self.outputs.get()
        s = self.memory_weights.get()
        W = self.weights.get()
        beta = self.beta

        ## TODO: make updates noisy? (perturbative)
        ## TODO: also, make a perturbation-based update synapse?
        if self.rule_fx == 1: ## contrastive (Movellan) Hebbian style plasticity
            ## TODO: add a loop to iterative over negative term several times
            ## we propagate the updated probe (negative) through memory to get a negative weighted state
            sims_hat = jnp.matmul(x_hat, W)
            s_hat = softmax(sims_hat - jnp.max(sims_hat, axis=1, keepdims=True) * beta) #s_hat = bkwta(s_hat, nWTA=1)
            ## positive Hebbian prod of probe+pos-state against negative Hebbian prod of updated-probe+neg-state
            term1 = (x.T @ s)
            term2 = -(x_hat.T @ s_hat)
            dW = term1 + term2
        #elif self.rule_fx == XX: ## deriv of energy w.r.t. memory W rule
        #    dW = x.T @ -s
        else: ## delta-rule (prescribed error rule) is the default
            dW = (x - x_hat).T @ s ## (deriv of MSE w.r.t. x_hat/updated probe)
        Ns = x.shape[0] ## get batch size
        dW = dW * (1./ Ns) ## we average batch updates

        ## TODO: add a term that checks if we need to append to memory W
        W = W + dW * self.eta - W * self.reg_lambda - jnp.sign(W) * self.l1_lambda ## actually adjust synaptic efficacies

        self.dWeights.set(dW)
        self.weights.set(W)

    @compilable
    def reset(self):
        inputVals = jnp.zeros((self.batch_size.get(), self.shape.get()[0]))
        outputVals = jnp.zeros((self.batch_size.get(), self.shape.get()[1]))

        if not self.inputs.targeted:
            self.inputs.set(inputVals)
        self.outputs.set(inputVals)
        self.similarities.set(outputVals)
        self.memory_weights.set(outputVals)
        self.energy.set(self.energy.get() * 0)
        self.dWeights.set(jnp.zeros(self.shape.get()))

    @classmethod
    def help(cls): ## component help function
        properties = {
            "synapse_type": "HopfieldSynapse - performs an adaptable synaptic transformation  of inputs to produce output "
                            "signals; synapses are adjusted via Hebbian learning in accordance with a Hopfield network"
        }
        compartment_props = {
            "input_compartments":
                {"inputs": "Takes in external input signal values",
                 "key": "JAX PRNG key"},
            "parameter_compartments":
                {"weights": "Synapse efficacy/strength parameter values"},
            "output_compartments":
                {"outputs": "Output of synaptic transformation"}
        }
        hyperparams = {
            "shape": "Shape of synaptic weight value matrix; number inputs x number outputs",
            "batch_size": "Batch size dimension of this component",
            "weight_init": "Initialization conditions for synaptic weight (W) values",
            "resist_scale": "Resistance level scaling factor (applied to output of transformation)",
            "p_conn": "Probability of a connection existing (otherwise, it is masked to zero)",
            "eta": "Global learning rate (to control update to memory matrix)",
            "beta": "Inverse temperature (controls softmax sharpness",
            "reg_lambda": "Weight decay coefficient to apply to local memory matrix updates",
            "update_rule": "What type of rule to use to update memory matrix (locally)",
            "initial_patterns": "Matrix containing a series of concatenated vectors to store into memory explicitly",
        }
        info = {cls.__name__: properties,
                "compartments": compartment_props,
                "dynamics": "outputs = Hopfield memory retrieval ;"
                            "dW = Hopfield Hebbian update",
                "hyperparameters": hyperparams}
        return info

# if __name__ == '__main__':
#     from ngcsimlib.context import Context
#     with Context("Bar") as bar:
#         Wab = HopfieldSynapse("Wab", (2, 3), 4, 4, 1.)
#     print(Wab)
