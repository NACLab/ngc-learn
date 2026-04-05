from jax import random, numpy as jnp, jit
from ngclearn import compilable #from ngcsimlib.parser import compilable
from ngclearn import Compartment #from ngcsimlib.compartment import Compartment
from ngclearn.utils.model_utils import softmax, bkwta #, chebyshev_norm

from ngclearn.components.synapses.denseSynapse import DenseSynapse

def _gaussian_kernel(dist, sigma): ## Gaussian weighting function
    density = jnp.exp(-jnp.power(dist, 2) / (2 * (sigma ** 2)))  # n_units x 1
    return density

class VectorQuantizeSynapse(DenseSynapse): # Vector quantization (VQ) synaptic cable
    """
    A synaptic cable that emulates a vector quantization memory model (the base case of this 
    model is referred to as "learning vector quantization"; LVQ). 

    | --- Synapse Compartments: ---
    | inputs - input (takes in external signals)
    | outputs - output signals (transformation induced by synapses)
    | weights - current value matrix of synaptic efficacies
    | bmu - current best-matching unit (BMU) mask, based on current inputs
    | i_tick - current internal tick / marker (gets incremented by 1 for each call to `evolve`)
    | eta - current learning rate value
    | key - JAX PRNG key
    | --- Synaptic Plasticity Compartments: ---
    | inputs - pre-synaptic signal/value to drive 1st term of VQ update (x)
    | outputs - post-synaptic signal/value to drive 2nd term of VQ update (y)
    | dWeights - current delta matrix containing changes to be applied to synapses

    | References:
    | Kohonen, Teuvo. "The self-organizing map." Proceedings of the IEEE 78.9 (2002): 1464-1480.

    Args:
        name: the string name of this cell

        shape: tuple specifying shape of this synaptic cable (usually a 2-tuple with number of 
            inputs by number of outputs) 

        eta: (initial) learning rate / step-size for this VQ model (initial condition value for `eta`)

        distance_function: tuple specifying distance function and its order for computing best-matching units (BMUs)
            (Default: ("minkowski", 2)).
            usage guide:
            ("minkowski", 2) or ("euclidean", ?) => use L2 norm (Euclidean) distance; 
            ("minkowski", 1) or ("manhattan", ?) => use L1 norm (taxi-cab/city-block) distance; 
            ("minkowksi", jnp.inf) or ("chebyshev", ?) => use Chebyshev distance; 
            ("minkowski", p > 2) => use a Minkowski distance of p-th order

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
            shape, ## determines codebook size
            eta=0.3, ## learning rate
            eta_decrement=0.00001, ## learning rate linear decrease (per update)
            syn_decay=0., ## weight decay term
            w_bound=0., 
            distance_function=("minkowski", 2),
            initial_patterns=None, ## possible class-based prototypes to init by
            weight_init=None,
            resist_scale=1.,
            p_conn=1.,
            batch_size=1,
            **kwargs
    ):
        super().__init__(
            name, shape, weight_init, None, resist_scale, p_conn, batch_size=batch_size, **kwargs
        )

        ### Synapse and VQ hyper-parameters
        self.K = 1 ## number of winners for a bmu
        dist_fun, dist_order = distance_function ## Default: ("minkowski", 2) -> Euclidean
        if "euclidean" in dist_fun.lower():
            dist_order = 2
        elif "manhattan" in dist_fun.lower(): 
            dist_order = 1
        elif "chebyshev" in dist_fun.lower():
            dist_order = jnp.inf
        ## TODO: add in cosine-distance (and maybe Mahalanobis distance)
        self.dist_order = dist_order ## set distance order p 

        self.shape = shape ## shape of synaptic efficacy matrix
        self.initial_eta = eta
        self.eta_decr = eta_decrement #0.001
        self.syn_decay = syn_decay
        self.w_bound = w_bound ## soft synaptic value bound (on magnitude)

        ## VQ Compartment setup
        self.eta = Compartment(jnp.zeros((1, 1)) + self.initial_eta)
        self.i_tick = Compartment(jnp.zeros((1, 1)))
        self.bmu = Compartment(jnp.zeros((1, 1)))
        #self.delta = Compartment(self.weights.get() * 0)
        self.dWeights = Compartment(self.weights.get() * 0)

    @compilable
    def advance_state(self): ## forward-inference step of VQ
        x_in = self.inputs.get()
        W = self.weights.get().T ## get (transposed) memory matrix

        ### We do some 3D tensor math to handle a batch of predictions that need to be made
        ### B = batch-size, D = embedding/input dim, C = number classes, N = number of memories
        _W = jnp.expand_dims(W, axis=0)  ## 3D tensor format of memory (1 x N x D)
        _x_in = jnp.expand_dims(x_in, axis=1)  ## 3D projection of input signals (B x 1 x D)
        D = _x_in - _W  ## compute 3D batched delta tensor (B x N x D)

        ## now apply distance function measuremnt over 3D tensor of deltas
        ## get batched (negative) distance measurements
        dist = jnp.linalg.norm(D, ord=self.dist_order, axis=2, keepdims=True) ## (B x N x 1)
        dist = -jnp.squeeze(dist, axis=2)  ## (B x N) (negative distance to find minimal vals)

        ## now get K winners per sample in batch
        #values, indices = lax.top_k(dist, K)
        bmu_mask = bkwta(dist, self.K)
        self.outputs.set(bmu_mask)

    @compilable
    def evolve(self, t, dt):  ## competitive Hebbian update step of VQ
        W = self.weights.get()
        x_in = self.inputs.get()
        z_out = self.outputs.get()
        tmp_key, *subkeys = random.split(self.key.get(), 3)
        self.key.set(tmp_key)
        ## synaptic update noise
        eps = random.normal(subkeys[0], W.shape) ## TODO: is this same size as tensor? or scalar?

        ## do the competitive Hebbian update
        dW = jnp.matmul(x_in.T, z_out) ## (N X D)
        #print("dW ", jnp.linalg.norm(dW))
        ## TODO: compute sign of dW given label match (-1 if no match, +1 if match)
        self.dWeights.set(dW)
        #print("W(t) ", jnp.linalg.norm(W))
        dW = dW * self.eta.get() - (W * self.syn_decay) ## inject weight decay
        #dW = dW + jnp.sqrt(2. * self.eta.get()) * eps ## inject Langevin noise
        zeta = 0.2 #0.35 #1. ## Langevin dampening factor
        dW = dW + eps * (2. * self.eta.get()) * zeta ## noise term (prevents going to zero in theory)
        if self.w_bound > 0.:
            ## enforce a soft value bound
            dW = dW * (self.w_bound - jnp.abs(W))
        ## else, do not apply soft-bounding
        W = W + dW * self.eta.get()
        self.weights.set(W)
        #print("W(t+1) ", jnp.linalg.norm(W))
        #exit()

        ## update learning rate alpha
        #a = self.eta.get()
        #a = a + (-a) * (1./self.tau_eta)
        eta_tp1 = jnp.maximum(1e-5, self.eta.get() - self.eta_decr)
        self.eta.set(eta_tp1)

        self.i_tick.set(self.i_tick.get() + 1)

    @compilable
    def reset(self):
        preVals = jnp.zeros((self.batch_size.get(), self.shape.get()[0]))
        postVals = jnp.zeros((self.batch_size.get(), self.shape.get()[1]))

        if not self.inputs.targeted:
            self.inputs.set(preVals)
        self.outputs.set(postVals)
        self.dWeights.set(jnp.zeros(self.shape.get()))
        #self.delta.set(jnp.zeros(self.shape.get()))
        self.bmu.set(self.bmu.get() * 0)
        #self.neighbor_weights.set(jnp.zeros((1, self.shape.get()[1])))

    @classmethod
    def help(cls): ## component help function
        properties = {
            "synapse_type": "VectorQuantizeSynapse - performs an adaptable synaptic transformation of inputs to produce output "
                            "signals; synapses are adjusted via competitive Hebbian learning in accordance with a "
                            "vector quantization model"
        }
        compartment_props = {
            "input_compartments":
                {"inputs": "Takes in external input signal values",
                 "key": "JAX PRNG key"},
            "parameter_compartments":
                {"weights": "Synapse efficacy/strength parameter values"},
            "output_compartments":
                {"outputs": "Output of synaptic transformation",
                 "bmu": "Best-matching unit (BMU) mask"},
        }
        hyperparams = {
            "shape": "Shape of synaptic weight value matrix; number inputs x number outputs",
            "batch_size": "Batch size dimension of this component",
            "weight_init": "Initialization conditions for synaptic weight (W) values",
            "resist_scale": "Resistance level scaling factor (applied to output of transformation)",
            "p_conn": "Probability of a connection existing (otherwise, it is masked to zero)",
            "eta": "Global learning rate",
            "distance_function": "Distance function tuple specifying how to compute BMUs"
        }
        info = {cls.__name__: properties,
                "compartments": compartment_props,
                "dynamics": "outputs = [bmu_mask] ;"
                            "dW = VQ competitive Hebbian update",
                "hyperparameters": hyperparams}
        return info

# if __name__ == '__main__':
#     from ngcsimlib.context import Context
#     with Context("Bar") as bar:
#         Wab = VectorQuantizeSynapse("Wab", (2, 3), 4, 4, 1.)
#     print(Wab)
