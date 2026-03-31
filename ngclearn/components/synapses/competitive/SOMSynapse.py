from jax import random, numpy as jnp, jit
from ngclearn import compilable #from ngcsimlib.parser import compilable
from ngclearn import Compartment #from ngcsimlib.compartment import Compartment
from ngclearn.utils.model_utils import softmax

from ngclearn.components.synapses.denseSynapse import DenseSynapse

def _gaussian_kernel(dist, sigma): ## Gaussian neighborhood function
    density = jnp.exp(-jnp.power(dist, 2) / (2 * (sigma ** 2)))  # n_units x 1
    return density

def _ricker_marr_kernel(dist, sigma): ## mexican hat neighborhood function
    ## can reform Ricker-Marr in terms of a function of a Gaussian density
    gauss_density = _gaussian_kernel(dist, sigma)
    density = gauss_density * (1. - (jnp.power(dist, 2) / (sigma ** 2)))
    # NOTE: Since the mexican hat density can produce negative values,
    #       we clip to 0 to avoid this as negative density messes up SOM learning
    return jnp.maximum(density, 0.)

def _euclidean_dist(a, b): ## Euclidean (L2) distance
    delta = a - b
    d = jnp.linalg.norm(delta, axis=0, keepdims=True)
    return d, delta

def _manhattan_dist(a, b): ## Manhattan (L1) distance
    delta = a - b
    d = jnp.linalg.norm(delta, ord=1, axis=0, keepdims=True)
    return d, delta

def _cosine_dist(a, b): ## Cosine-similarity distance
    delta = a - b
    d = 1. - (jnp.matmul(a.T, b) / (jnp.linalg.norm(a, axis=0) * jnp.linalg.norm(b, axis=0)))
    return d, delta

class SOMSynapse(DenseSynapse): # Self-organizing map (SOM) synaptic cable
    """
    A synaptic cable that emulates a self-organizing map (or Kohonen map) that is adapted via
    competitive Hebbian learning. Many of this synapses internal compartments house dynamically-updated
    values for learning elements such as the SOM's neighborhood radius and learning rate.

    Mathematically, a synaptic update performed according to SOM theory is:
    | Delta W_{ij} = (x.T - W) * n(BMU) * eta
    | where n(BMU) is a neighborhood weighting function centered around (topological) coordinates of BMU
    | where x is vector of pre-synaptic inputs, W is SOM's synaptic matrix, and BMU is best-matching unit for x

    | --- Synapse Compartments: ---
    | inputs - input (takes in external signals)
    | outputs - output signals (transformation induced by synapses)
    | weights - current value matrix of synaptic efficacies
    | bmu - current best-matching unit (BMU), based on current inputs
    | delta - current differences between inputs and each weight vector of this SOM's synaptic matrix
    | i_tick - current internal tick / marker (gets incremented by 1 for each call to `evolve`)
    | eta - current learning rate value
    | radius - current radius value to control neighborhood function
    | key - JAX PRNG key
    | --- Synaptic Plasticity Compartments: ---
    | inputs - pre-synaptic signal/value to drive 1st term of SOM update (x)
    | outputs - post-synaptic signal/value to drive 2nd term of SOM update (y)
    | neighbor_weights - topology weighting applied to synaptic adjustments
    | dWeights - current delta matrix containing changes to be applied to synapses

    | References:
    | Kohonen, Teuvo. "The self-organizing map." Proceedings of the IEEE 78.9 (2002): 1464-1480.

    Args:
        name: the string name of this cell

        n_inputs: number of input units to this SOM

        n_units_x: number of output units along length of rectangular topology of this SOM

        n_units_y:  number of output units along width of rectangular topology of this SOM

        eta: (initial) learning rate / step-size for this SOM (initial condition value for `eta`)

        distance_function: string specifying distance function to use for finding best-matching units (BMUs)
            (Default: "euclidean").
            usage guide:
            "euclidean" = use L2 / Euclidean distance
            "manhattan" = use L1 / Manhattan / taxi-cab distance
            "cosine" = use cosine-similarity distance

        neighbor_function: string specifying neighborhood function to compute approximate topology weighting across
            units in topology (based on BMU) (Default: "gaussian").
            usage guide:
            "gaussian" = use Gaussian kernel
            "ricker" = use Mexican-hat / Ricker-Marr kernel

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
            n_inputs,
            n_units_x, ## num units along width of SOM rectangular topology
            n_units_y, ## num units along length of SOM rectangular topology
            eta=0.5, ## learning rate
            distance_function="euclidean",
            neighbor_function="gaussian",
            weight_init=None,
            resist_scale=1.,
            p_conn=1.,
            batch_size=1,
            **kwargs
    ):
        shape = (n_inputs, n_units_x * n_units_y)
        super().__init__(name, shape, weight_init, None, resist_scale, p_conn, batch_size=batch_size, **kwargs)

        ### build (rectangular) topology coordinates
        coords = []
        for i in range(n_units_x):
            x = jnp.ones((n_units_x, 1)) * i
            y = jnp.expand_dims(jnp.arange(start=0, stop=n_units_y), axis=1)
            xy = jnp.concat((x, y), axis=1)
            coords.append(xy)
        self.coords = jnp.concat(coords, axis=0)

        ### Synapse and SOM hyper-parameters
        #self.radius = radius
        self.distance_function = distance_function
        self.dist_fx = 0 ## default := 0 (euclidean)
        if "manhattan" in distance_function:
            self.dist_fx = 1
        elif "cosine" in distance_function:
            self.dist_fx = 2
        self.neighbor_function = neighbor_function
        self.neighbor_fx = 0 ## default := 0 (Gaussian)
        if "ricker" in neighbor_function:
            self.neighbor_fx = 1 ## Mexican-hat function
        self.shape = shape ## shape of synaptic efficacy matrix

        ## exponential decay -> dz/dt = -kz has sol'n:  z0 exp(-k t)
        # self.iterations = 50000
        # self.initial_eta = eta ## alpha (in SOM-lingo) #0.5
        # self.initial_radius = jnp.maximum(n_units_x, n_units_y) / 2  #n_units_x / 2
        # self.C = self.iterations / jnp.log(self.initial_radius)

        ## exponential decay -> dz/dt = -kz has sol'n:  z0 exp(-k t)
        self.initial_eta = eta  ## alpha (in SOM-lingo) #0.5
        self.initial_radius = jnp.maximum(n_units_x, n_units_y) / 2
        self.tau_eta = 50000
        self.tau_radius = self.tau_eta / jnp.log(self.initial_radius) ## C

        ## SOM Compartment setup
        self.radius = Compartment(jnp.zeros((1, 1)) + self.initial_radius)
        self.eta = Compartment(jnp.zeros((1, 1)) + self.initial_eta)
        self.i_tick = Compartment(jnp.zeros((1, 1)))
        self.bmu = Compartment(jnp.zeros((1, 1)))
        self.delta = Compartment(self.weights.get() * 0)
        self.neighbor_weights = Compartment(jnp.zeros((1, shape[1])))
        self.dWeights = Compartment(self.weights.get() * 0)

    def _calc_bmu(self): ## obtain index of best-matching unit (BMU)
        x = self.inputs.get()
        W = self.weights.get()
        # W * I - x * I ?
        if self.dist_fx == 1:  ## L1 distance
            d, delta = _manhattan_dist(x.T, W)
        elif self.dist_fx == 2: ## cosine distance
            d, delta = _cosine_dist(x.T, W)
        else: ## L2 distance
            d, delta = _euclidean_dist(x.T, W)
        bmu = jnp.argmin(d, axis=1, keepdims=True)
        bmu_idx = bmu #bmu[0, 0]
        return bmu_idx, delta

    def _calc_neighborhood_weights(self):  ## neighborhood function
        bmu = self.bmu.get()[0, 0] ## get best-matching unit
        coords = self.coords ## constant coordinate array
        radius = self.radius.get() ## get current neighborhood radius value
        coord_bmu = coords[bmu:bmu + 1, :]  ## TODO: might need to one-hot mask + sum
        delta = coords - coord_bmu  ## raw coordinate differences (delta)

        ### neighborhood-weighting computation note: 
        ### internally, calculation of neighborhood weighting depends on 1st calculating
        ### L2 distance in Cartesian coordinate-space, then applying the neighborhood 
        ### over these coordinate distance values
        bmu_dist = jnp.linalg.norm(delta, axis=1, keepdims=True)
        if self.neighbor_fx == 1: ## apply Mexican-hat kernel
            neighbor_weights = _ricker_marr_kernel(bmu_dist, sigma=radius)
        else: ## apply Gaussian kernel
            neighbor_weights = _gaussian_kernel(bmu_dist, sigma=radius)
        ## TODO: add in triangular, bubble, & laplacian kernels

        return neighbor_weights.T  ## transpose to (1 x n_units)

    @compilable
    def advance_state(self): ## forward-inference step of SOM
        bmu_idx, delta = self._calc_bmu()
        self.bmu.set(bmu_idx) ## store BMU
        self.delta.set(delta) ## store delta/differences
        neighbor_weights = self._calc_neighborhood_weights()
        self.neighbor_weights.set(neighbor_weights) ## store neighborhood weightings

        ## compute an approximate weighted activity output for input pattern
        #activity = jnp.sum(self.weights * self.resist_scale * neighbor_weights, axis=1, keepdims=True)
        ### obtain weighted competitive activations (via softmax probs)
        activity = softmax(neighbor_weights * self.resist_scale)
        self.outputs.set(activity)

    @compilable
    def evolve(self, t, dt):  ## competitive Hebbian update step of SOM
        #bmu = self.bmu.get() ## best-matching unit
        delta = self.delta.get() ## deltas/differences between input & all SOM templates
        neighbor_weights = self.neighbor_weights.get() ## get neighborhood weight values

        ## exponential decay -> dz/dt = -kz has sol'n:  z0 exp(-k t)
        #t = self.i_tick.get()
        ## update radius
        r = self.radius.get()
        r = r + (-r) * (1./self.tau_radius)
        self.radius.set(r)
        ## update learning rate alpha
        a = self.eta.get()
        a = a + (-a) * (1./self.tau_eta)
        self.eta.set(a)
        # self.radius.set(self.initial_radius * jnp.exp(-self.i_tick.get() / self.C))  ## update radius
        # self.eta.set(self.initial_eta * jnp.exp(-self.i_tick.get() / self.iterations)) ## update learning rate alpha

        dWeights = delta * neighbor_weights * self.eta.get() ## calculate change-in-synapses
        self.dWeights.set(dWeights)
        _W = self.weights.get() + dWeights ## update via competitive Hebbian rule
        self.weights.set(_W)

        self.i_tick.set(self.i_tick.get() + 1)

    @compilable
    def reset(self):
        preVals = jnp.zeros((self.batch_size.get(), self.shape.get()[0]))
        postVals = jnp.zeros((self.batch_size.get(), self.shape.get()[1]))

        if not self.inputs.targeted:
            self.inputs.set(preVals)
        self.outputs.set(postVals)
        self.dWeights.set(jnp.zeros(self.shape.get()))
        self.delta.set(jnp.zeros(self.shape.get()))
        self.bmu.set(jnp.zeros((1, 1)))
        self.neighbor_weights.set(jnp.zeros((1, self.shape.get()[1])))

    @classmethod
    def help(cls): ## component help function
        properties = {
            "synapse_type": "SOMSynapse - performs an adaptable synaptic transformation  of inputs to produce output "
                            "signals; synapses are adjusted via competitive Hebbian learning in accordance with a "
                            "Kohonen map"
        }
        compartment_props = {
            "input_compartments":
                {"inputs": "Takes in external input signal values",
                 "key": "JAX PRNG key"},
            "parameter_compartments":
                {"weights": "Synapse efficacy/strength parameter values"},
            "output_compartments":
                {"outputs": "Output of synaptic transformation",
                 "bmu": "Best-matching unit (BMU)"},
        }
        hyperparams = {
            "shape": "Shape of synaptic weight value matrix; number inputs x number outputs",
            "batch_size": "Batch size dimension of this component",
            "weight_init": "Initialization conditions for synaptic weight (W) values",
            "resist_scale": "Resistance level scaling factor (applied to output of transformation)",
            "p_conn": "Probability of a connection existing (otherwise, it is masked to zero)",
            "eta": "Global learning rate",
            "radius": "Radius parameter to control influence of neighborhood function",
            "distance_function": "Distance function used to compute BMU"
        }
        info = {cls.__name__: properties,
                "compartments": compartment_props,
                "dynamics": "outputs = [W * alpha(bmu)] ;"
                            "dW = SOM competitive Hebbian update",
                "hyperparameters": hyperparams}
        return info

# if __name__ == '__main__':
#     from ngcsimlib.context import Context
#     with Context("Bar") as bar:
#         Wab = SOMSynapse("Wab", (2, 3), 4, 4, 1.)
#     print(Wab)
