from jax import random, numpy as jnp, jit, nn
from functools import partial
from ngclearn import compilable 
from ngclearn import Compartment 
from ngclearn.utils.model_utils import softmax, bkwta

from ngclearn.components.synapses.denseSynapse import DenseSynapse

@partial(jit, static_argnums=[1])
def _normalize(x_in, norm_fx=0):
    if norm_fx == 1:
        xmin = jnp.min(x, axis=1, keepdims=True)
        xmax = jnp.max(x, axis=1, keepdims=True)
        x = (x_in - xmin)/(xmax - xmin)
    else:
        x = x_in / jnp.linalg.norm(x_in, ord=2, axis=1, keepdims=True)
    return x

class ART2ASynapse(DenseSynapse): # Adaptive resonance theory (ART) 2A synaptic cable
    """
    A synaptic cable that emulates a simplified form of adaptive resonance theory (ART) 
    adapted for continuous input signals (specifically, the ART2A-C model that handles 
    real-valued input values).

    | --- Synapse Compartments: ---
    | inputs - input (takes in external signals)
    | outputs - output signals (transformation induced by synapses)
    | weights - current value matrix of synaptic efficacies
    | i_tick - current internal tick / marker (gets incremented by 1 for each call to `evolve`)
    | eta - current learning rate value
    | key - JAX PRNG key
    | --- Synaptic Plasticity Compartments: ---
    | inputs - pre-synaptic signal/value to drive 1st term of ART2A update (x)
    | outputs - post-synaptic signal/value to drive 2nd term of ART2A update (y)
    | dWeights - current delta matrix containing changes to be applied to synapses

    | References:
    | Carpenter, Gail A., and Stephen Grossberg. "ART 2: Self-organization of stable category 
    | recognition codes for analog input patterns." Applied optics 26.23 (1987): 4919-4930.
    |
    | Ororbia, Alexander G. "Continual competitive memory: A neural system for online task-free 
    | lifelong learning." arXiv preprint arXiv:2106.13300 (2021).

    Args:
        name: the string name of this cell

        shape: tuple specifying shape of this synaptic cable (usually a 2-tuple with number of 
            inputs by number of outputs) 

        eta: (initial) learning rate / step-size for this ART2A model (initial condition value for `eta`)

        eta_decrement: constant value to decrease `eta` by each call to this synapse's `evolve()`, i.e., 
            this triggers a linear schedule for decreasing `eta` by (Default: 0)

        vigilance: vigilance parameter to decide if a memory vector is updated (rho)

        weight_init: a kernel to drive initialization of this synaptic cable's values;
            typically a tuple with 1st element as a string calling the name of
            initialization to use

        resist_scale: a fixed scaling factor to apply to synaptic transform (Default: 1.)

        p_conn: probability of a connection existing (default: 1.); setting
            this to < 1. will result in a sparser synaptic structure
    """

    def __init__(
            self,
            name,
            shape, ## determines memory matrix size
            eta=0.05, ## learning rate
            eta_decrement=0., ## linear scheduled decrement over eta
            vigilance=0.3, ## vigilance parameter (rho)
            weight_init=None,
            resist_scale=1.,
            p_conn=1.,
            batch_size=1,
            **kwargs
    ):
        super().__init__(
            name, shape, weight_init, None, resist_scale, p_conn, batch_size=batch_size, **kwargs
        )

        ### Synapse and ART-2A hyper-parameters
        self.K = 1 ## number of winners for bmu calculation
        self.norm_fx = 0 ## 0 -> normalize via norm, 1 -> complement coding (min-max rescale)
       
        self.shape = shape ## shape of synaptic efficacy matrix
        self.initial_eta = eta
        self.eta_decr = eta_decrement ## linear decrease to iteratively update eta by (each "tick")
        self.vigilance = vigilance ## (rho)

        ## ART-2A Compartment setup
        self.xprobe = Compartment(jnp.zeros((batch_size, shape[0])))
        self.eta = Compartment(jnp.zeros((1, 1)) + self.initial_eta, display_name="Dynamic step size")
        self.i_tick = Compartment(jnp.zeros((1, 1)))
        #self.bmu = Compartment(jnp.zeros((1, 1)), display_name="Best matching unit mask")
        self.dWeights = Compartment(self.weights.get() * 0)
        self.misses = Compartment(jnp.zeros((batch_size, 1))) ## marker for non-resonant patterns in a batch

        self.weights.set(self.weights.get() * 0)
        self.used = Compartment(jnp.zeros((1, shape[1]))) ## marks if memory slot used

    def insert(self, x, idx): ## manual memory insertion co-routine
        W = self.weights.get()
        z_m = jnp.expand_dims(nn.one_hot(idx, W.shape[1]), axis=0)
        dW = (W * 0 + x.T) * z_m
        W = W + dW
        self.weights.set(W)
        self.used.set(((self.used.get() + z_m) > 0.) * 1.)

    def grow(self, n_memories): ## grow out memory matrix by fixed amount
        W = self.weights.get()
        used = self.used.get()
        ## expand memory matrix by a fixed set of empty memory slots
        W = jnp.concat([W, jnp.zeros((W.shape[0], n_memories))], axis=1)
        n_unused = jnp.zeros((1, n_memories))
        used = jnp.concat([used, n_unused], axis=1)
        #print("used: ", used.shape)
        self.used.set(used)
        self.weights.set(W)
        self.dWeights.set(W * 0)
        self.shape = self.weights.get().shape

    @compilable
    def consolidate(self): ## memory consolition co-routine (for non-resonant signals)
        n_used = int(jnp.sum(self.used.get())) ## number unused slots left
        x = self.xprobe.get()
        W = self.weights.get()
        nonresonants = self.misses.get()
        ## we project non-resonant memories to empty slots in memory W
        S = jnp.eye(x.shape[0], self.shape[1], k=n_used)
        dWstore = jnp.matmul((x * nonresonants).T, S)
        W = W + dWstore ## Hebbian update to memory
        ## re-compute number of used slots post-consolidation
        nW = jnp.linalg.norm(W, ord=2, axis=0, keepdims=True)
        used = (nW > 0.) * 1

        self.weights.set(W)
        self.used.set(used)
        ## else, currently discard un-absorbed/non-resonant patterns
        ##       can add a function that "grows" out block matrix by a chunk (to control growth)
        ## TODO: add pruning mechanism for low-usage slots

    @compilable
    def advance_state(self): ## forward-inference step of ART2A
        x_in = self.inputs.get()
        W = self.weights.get() ## get (transposed) memory matrix
        used = self.used.get()

        x = _normalize(x_in, norm_fx=self.norm_fx) 
        self.xprobe.set(x)
        sims = jnp.matmul(x, W) ## compute similarities (parallel dot products)
        ## we correct activities by masking out unused slots
        sims_min = jnp.amin(sims, axis=1, keepdims=True)
        sims = sims * used + (1. - used) * (sims_min - 1.)
        z_winners = sims * bkwta(sims, nWTA=self.K) ## get winner mask (hidden layer)
        self.outputs.set(z_winners)

    @compilable
    def evolve(self, t, dt):  ## competitive Hebbian update step of ART2A
        W = self.weights.get() ## D x Z
        x = self.xprobe.get() ## B x D
        z_winners = self.outputs.get() ## B x Z
        eta = self.eta.get()
        ## Note: we refactor ART update into a leaky integrator equation:
        ## W = W * (1 - b) + dW * b = W + b * (-W + dW); b = eta
        
        ## for resonant patterns, we perform a Hebbian storage update
        hits = (z_winners >= self.vigilance) * 1. ## B x Z
        m = (jnp.sum(hits, axis=1, keepdims=True) > 0.) * 1. ## B x 1
        wnew = (-jnp.matmul(z_winners, W.T) + x) * m ## B x D
        dW = jnp.matmul(wnew.T, hits) ## D x Z ## adjustment matrix
        W = W + dW * eta ## D x Z ## do a step of Hebbian ascent
        nonresonants = 1. - m ## mark non-resonant patterns in batch

        ## NOTE: is this post-weight-update normalization needed?
        #nW = jnp.linalg.norm(W, ord=2, axis=0, keepdims=True)
        #used = (nW > 0.) * 1
        #mz = (jnp.sum(hits, axis=0, keepdims=True) > 0.) * 1.
        #W = W / (nW * mz + (1. - mz))

        self.weights.set(W)
        self.misses.set(nonresonants) ## store unused/non-resonant pattern mas
    
        #tmp_key, *subkeys = random.split(self.key.get(), 3)
        #self.key.set(tmp_key)
        ## synaptic update noise
        #eps = random.normal(subkeys[0], W.shape) ## TODO: is this same size as tensor? or scalar?

        ## update learning rate eta
        eta_tp1 = jnp.maximum(1e-5, eta - self.eta_decr)
        self.eta.set(eta_tp1)

        self.i_tick.set(self.i_tick.get() + 1)

    @compilable
    def reset(self):
        preVals = jnp.zeros((self.batch_size.get(), self.shape.get()[0]))
        postVals = jnp.zeros((self.batch_size.get(), self.shape.get()[1]))

        if not self.inputs.targeted:
            self.inputs.set(preVals)
        self.outputs.set(postVals)
        self.xprobe.set(preVals)
        #self.misses.set(jnp.zeros((self.batch_size.get(), 1)))
        self.misses.set(self.misses.get() * 0)
        self.dWeights.set(jnp.zeros(self.shape.get()))

    @classmethod
    def help(cls): ## component help function
        properties = {
            "synapse_type": "ART2ASynapse - performs an adaptable synaptic transformation of inputs to produce output "
                            "signals; synapses are adjusted via competitive Hebbian learning in accordance with "
                            "adaptive resonance theory (2A)"
        }
        compartment_props = {
            "input_compartments":
                {"inputs": "Takes in external input signal values",
                 "key": "JAX PRNG key"},
            "parameter_compartments":
                {"weights": "Synapse efficacy/strength parameter values"},
            "output_compartments":
                {"outputs": "Output of synaptic transformation"},
        }
        hyperparams = {
            "shape": "Shape of synaptic weight value matrix; number inputs x number outputs",
            "batch_size": "Batch size dimension of this component",
            "weight_init": "Initialization conditions for synaptic weight (W) values",
            "resist_scale": "Resistance level scaling factor (applied to output of transformation)",
            "p_conn": "Probability of a connection existing (otherwise, it is masked to zero)",
            "eta": "Global learning rate",
            "eta_decrement": "Constant amount to decrease global learning by each call to `evolve`"
        }
        info = {cls.__name__: properties,
                "compartments": compartment_props,
                "dynamics": "outputs = [bmu_mask] ;"
                            "dW = ART2A competitive Hebbian update",
                "hyperparameters": hyperparams}
        return info

# if __name__ == '__main__':
#     from ngcsimlib.context import Context
#     with Context("Bar") as bar:
#         Wab = ART2ASynapse("Wab", (2, 3), 4, 4, 1.)
#     print(Wab)
