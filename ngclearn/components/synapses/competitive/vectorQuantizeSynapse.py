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
    | labels - label input (optional)
    | outputs - output signals (transformation induced by synapses)
    | weights - current value matrix of synaptic efficacies
    | label_weights - current value of matrix label efficacies (if this VQ is supervised)
    | i_tick - current internal tick / marker (gets incremented by 1 for each call to `evolve`)
    | eta - current learning rate value
    | key - JAX PRNG key
    | --- Synaptic Plasticity Compartments: ---
    | inputs - pre-synaptic signal/value to drive 1st term of VQ update (x)
    | outputs - post-synaptic signal/value to drive 2nd term of VQ update (y)
    | labels - (optional) pre-synaptic signal to drive 1st term of VQ update to label matrix
    | dWeights - current delta matrix containing changes to be applied to synapses

    | References:
    | Somervuo, Panu, and Teuvo Kohonen. "Self-organizing maps and learning vector quantization for 
    | feature sequences." Neural Processing Letters 10.2 (1999): 151-159.
    |
    | Kohonen, Teuvo. "The self-organizing map." Proceedings of the IEEE 78.9 (2002): 1464-1480.
    |
    | Ororbia, Alexander G. "Continual competitive memory: A neural system for online task-free 
    | lifelong learning." arXiv preprint arXiv:2106.13300 (2021).

    Args:
        name: the string name of this cell

        shape: tuple specifying shape of this synaptic cable (usually a 2-tuple with number of 
            inputs by number of outputs) 

        eta: (initial) learning rate / step-size for this VQ model (initial condition value for `eta`)

        eta_decrement: a constant value to linearly decrease `eta` by per synaptic update 
            (Default: 0, which disables this)

        syn_decay: a synaptic weight (L2) decay to apply to synapses per update

        w_bound: upper soft bound to enforce over synapses post-update (Default: 0, which disables this scaling term)  

        distance_function: tuple specifying distance function and its order for computing best-matching units (BMUs)
            (Default: ("minkowski", 2)).
            usage guide:
            ("minkowski", 2) or ("euclidean", ?) => use L2 norm (Euclidean) distance; 
            ("minkowski", 1) or ("manhattan", ?) => use L1 norm (taxi-cab/city-block) distance; 
            ("minkowksi", jnp.inf) or ("chebyshev", ?) => use Chebyshev distance; 
            ("minkowski", p > 2) => use a Minkowski distance of p-th order

        label_dim: dimensionality of label neurons (corresponding to each memory/prototype); note this is 
            inactive/unused if <= 0 (Default: 0)

        initial_patterns: a tuple containing data vectors (and labels) to initialize code-book by; 
            note if `label_dim` <= 0, then only first element of tuple will be used (Default: None)

        langevin_noise_scale: scale factor to control degree to which Langevin sampling noise is 
            applied to a given synaptic weight update (Default: 0, which disables this)

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
            label_dim=0, ## if > 0, then this becomes supervised LVQ(1)
            initial_patterns=None, ## possible class-based prototypes to init by
            lanvegin_noise_scale=0., ## scale of Langevin noise to apply to updates
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
        self.label_dim = label_dim
        self.K = 1 ## number of winners (for a bmu) 
        dist_fun, dist_order = distance_function ## Default: ("minkowski", 2) -> Euclidean
        if "euclidean" in dist_fun.lower():
            dist_order = 2
        elif "manhattan" in dist_fun.lower(): 
            dist_order = 1
        elif "chebyshev" in dist_fun.lower():
            dist_order = jnp.inf
        self.dist_order = dist_order ## set distance order p

        self.shape = shape ## shape of synaptic efficacy matrix
        self.initial_eta = eta
        self.eta_decr = eta_decrement #0.001
        self.syn_decay = syn_decay
        self.w_bound = w_bound ## soft synaptic value bound (on magnitude)
        self.zeta = langevin_noise_scale #0.2 #0.35 #1. ## Langevin dampening factor

        ## VQ Compartment setup
        label_syn_init = labels_init = jnp.zeros((1, 1))
        if self.label_dim > 0:
            label_syn_init = jnp.zeros((label_dim, self.shape[1]))
            labels_init = jnp.zeros((self.batch_size, self.label_dim))
        self.labels = Compartment(labels_init, display_name="Label Units")
        self.pred_labels = Compartment(labels_init, display_name="Predicted Label Values")
        self.label_weights = Compartment(label_syn_init, display_name="Label Synapses / Memory") 
        self.eta = Compartment(jnp.zeros((1, 1)) + self.initial_eta, display_name="Dynamic step size")
        self.i_tick = Compartment(jnp.zeros((1, 1)))
        #self.bmu = Compartment(jnp.zeros((1, 1)), display_name="Best matching unit mask")
        self.dWeights = Compartment(self.weights.get() * 0)

        if initial_patterns is not None: ## preload memory synaptic matrix
            initX, initY = initial_patterns
            W = self.weights.get()
            D, H = W.shape
            tmp_key, *subkeys = random.split(self.key.get(), 3)
            if initX.shape[1] < H: ## randomly portions of memory with stored patterns/templates
                ptrs = random.permutation(subkeys[0], H)
                W = jnp.concat([initX, W[:, 0:(H - initX.shape[1])]], axis=1)
                W = W[:, ptrs] ## shuffle memories
                self.weights.set(W)
                if self.label_dim > 0:
                    Wy = self.label_weights.get()
                    Wy = jnp.concat([initY, Wy[:, 0:(H - initX.shape[1])]], axis=1)
                    Wy = Wy[:, ptrs]  ## shuffle memories
                    self.label_weights.set(Wy)
            else: ## memory is exactly the set of stored patterns/templates
                self.weights.set(initX)
                if self.label_dim > 0:
                    self.label_weights.set(initY)
    @compilable
    def advance_state(self): ## forward-inference step of VQ
        x_in = self.inputs.get()
        x_in = x_in / jnp.linalg.norm(x_in, axis=1, keepdims=True)
        self.inputs.set(x_in)
        W = self.weights.get().T ## get (transposed) memory matrix

        ### We do some 3D tensor math to handle a batch of predictions that need to be made
        ### B = batch-size, D = embedding/input dim, C = number classes, N = number of memories
        _W = jnp.expand_dims(W, axis=0)  ## 3D tensor format of memory (1 x N x C)
        _x_in = jnp.expand_dims(x_in, axis=1)  ## 3D projection of input signals (B x 1 x D)
        D = _x_in - _W  ## compute 3D batched delta tensor (B x N x D)

        ## now apply distance function measuremnt over 3D tensor of deltas
        ## get batched (negative) distance measurements
        dist = jnp.linalg.norm(D, ord=self.dist_order, axis=2, keepdims=True) ## (B x N x 1)
        dist = -jnp.squeeze(dist, axis=2)  ## (B x N) (negative distance to find minimal vals)

        ## now get K winners per sample in batch
        #values, indices = lax.top_k(dist, K)
        bmu_mask = bkwta(dist, nWTA=self.K)
        self.outputs.set(bmu_mask)
        if self.label_dim > 0: ## store a label prediction (if applicable)
            pred_labels = jnp.matmul(bmu_mask, self.label_weights.get().T)
            self.pred_labels.set(pred_labels)

    @compilable
    def evolve(self, t, dt):  ## competitive Hebbian update step of VQ
        W = self.weights.get()
        x_in = self.inputs.get()
        z_out = self.outputs.get()
        
        ## do the competitive Hebbian update
        signed_x_in = x_in
        if self.label_dim > 0: ## first, compute the sign of the update if labels are available
            ## LVQ(1) => compute sign of dW given label match (-1 if no match, +1 if match)
            y_in = self.labels.get()
            YW = self.label_weights.get()#.T 
            y_mem = jnp.matmul(z_out, YW.T) ## decode to get label memories
            ## each row of `y_exists`: 1 if lab mem stored, 0 otherwise
            y_exists = (jnp.sum(y_mem, axis=1, keepdims=True) > 0.) * 1. 
            ## TODO: update YW with labels for initial cond?

            y_in_l = jnp.argmax(y_in, axis=1, keepdims=True) ## get lab indices
            y_mem_l = jnp.argmax(y_mem, axis=1, keepdims=True) ## get mem lab indices
            ## each of `dy`: -1 => incorrect (push away), +1 => correct (push towards)
            dy = (y_in_l == y_mem_l) * 2. - 1.
            dy = dy * y_exists + (1. - y_exists) ## +1 for each "empty" memory
            signed_x_in = x_in * dy ## sign (-1, +1) each update
        ## else, sign of all updates is +1
        
        ## second, given the above sign, compute the Hebbian adjustment and optional terms
        dW = jnp.matmul(signed_x_in.T, z_out) ## calc competitive Hebbian update (N x D)
        dW = dW - (W * self.syn_decay) ## inject weight decay

        if self.zeta > 0.: ## synaptic update noise
            tmp_key, *subkeys = random.split(self.key.get(), 3)
            self.key.set(tmp_key)
            eps = random.normal(subkeys[0], W.shape)
            dW = dW + jnp.sqrt(2. * self.eta.get()) * eps ## inject Langevin noise
            dW = dW + eps * (2. * self.eta.get()) * self.zeta ## noise term (prevents going to zero in theory)
        
        if self.w_bound > 0.: ## enforce a soft value bound
            dW = dW * (self.w_bound - jnp.abs(W))
        # ## else, do not apply soft-bounding
        self.dWeights.set(dW)

        ## third, apply the synaptic update to memory matrix W
        W = W + dW * self.eta.get()
        self.weights.set(W)

        ## update learning rate (eta)
        eta_tp1 = jnp.maximum(1e-5, self.eta.get() - self.eta_decr)
        self.eta.set(eta_tp1)

        self.i_tick.set(self.i_tick.get() + 1) ## advance internal "tick"

    @compilable
    def reset(self):
        preVals = jnp.zeros((self.batch_size.get(), self.shape.get()[0]))
        postVals = jnp.zeros((self.batch_size.get(), self.shape.get()[1]))

        if not self.inputs.targeted:
            self.inputs.set(preVals)
        self.outputs.set(postVals)
        self.labels.set(self.labels.get() * 0)
        self.pred_labels.set(self.pred_labels.get() * 0)
        self.dWeights.set(jnp.zeros(self.shape.get()))
        #self.bmu.set(self.bmu.get() * 0)

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
                 "labels": "Takes in (optional) label signal values", 
                 "key": "JAX PRNG key"},
            "parameter_compartments":
                {"weights": "Synapse efficacy/strength parameter values", 
                 "label_weights": "Label efficacy parameter values (if this VQ is supervised)"},
            "output_compartments":
                {"outputs": "Output of synaptic transformation", 
                 "pred_labels": "Predicted labels (if this VQ is supervised)"},
        }
        hyperparams = {
            "shape": "Shape of synaptic weight value matrix; number inputs x number outputs",
            "label_dim": "Dimensionality of labels (if this VQ is supervised)", 
            "batch_size": "Batch size dimension of this component",
            "weight_init": "Initialization conditions for synaptic weight (W) values",
            "resist_scale": "Resistance level scaling factor (applied to output of transformation)",
            "p_conn": "Probability of a connection existing (otherwise, it is masked to zero)",
            "eta": "Global learning rate",
            "eta_decrement": "Constant to decrement `eta` by per update/call to `evolve()`",
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
