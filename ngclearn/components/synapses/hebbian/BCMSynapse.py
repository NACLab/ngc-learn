from jax import random, numpy as jnp, jit
from ngclearn import compilable #from ngcsimlib.parser import compilable
from ngclearn import Compartment #from ngcsimlib.compartment import Compartment

from ngclearn.components.synapses.denseSynapse import DenseSynapse

class BCMSynapse(DenseSynapse): # BCM-adjusted synaptic cable
    """
    A synaptic cable that adjusts its efficacies in accordance with BCM
    (Bienenstock-Cooper-Munro) theory.

    Mathematically, a synaptic update performed according to BCM theory is:
    | tau_w d(W_{ij})/dt = -w_decay W_{ij} + x_j * [y_i * (y_i - theta_i)] / theta_i
    | tau_theta d(theta_i)/dt = -theta_i + <(y_i)^2>_{batch}
    | where x_j is the pre-synaptic input, y_i is the post-synaptic output

    Note that, in most literature related to BCM, the average value used for
    threshold `theta` can be assumed to be the average over all input patterns
    (as in a full dataset batch update) but a temporal average maintained for
    `theta` will "usually be equivalent" (and ngc-learn implements the threshold
    `theta` in terms of a leaky ODE to dynamically compute the temporal mean).

    | --- Synapse Compartments: ---
    | inputs - input (takes in external signals)
    | outputs - output signals (transformation induced by synapses)
    | weights - current value matrix of synaptic efficacies
    | key - JAX PRNG key
    | --- Synaptic Plasticity Compartments: ---
    | pre - pre-synaptic signal/value to drive 1st term of BCM update (x)
    | post - post-synaptic signal/value to drive 2nd term of BCM update (y)
    | theta - synaptic modification threshold (post-synaptic) variables
    | dWeights - current delta matrix containing changes to be applied to synapses

    | References:
    | Bienenstock, E. L., Cooper, L. N, and Munro, P. W. (1982). Theory for the
    | development of neuron selectivity: orientation specificity and binocular
    | interaction in visual cortex. Journal of Neuroscience, 2:32–48.

    Args:
        name: the string name of this cell

        shape: tuple specifying shape of this synaptic cable (usually a 2-tuple
            with number of inputs by number of outputs)

        tau_w: synaptic update time constant

        tau_theta: threshold variable evolution time constant

        theta0: initial condition for synaptic modification threshold

        w_bound: maximum value to enforce over newly computed efficacies
            (default: 0.); must > 0. to be used

        w_decay: synaptic decay factor (default: 0.)

        weight_init: a kernel to drive initialization of this synaptic cable's values;
            typically a tuple with 1st element as a string calling the name of
            initialization to use

        resist_scale: a fixed scaling factor to apply to synaptic transform
            (Default: 1.), i.e., yields: out = ((W * Rscale) * in)

        p_conn: probability of a connection existing (default: 1.); setting
            this to < 1. will result in a sparser synaptic structure
    """

    def __init__(
            self, name, shape, tau_w, tau_theta, theta0=-1., w_bound=0., w_decay=0., weight_init=None, resist_scale=1.,
            p_conn=1., batch_size=1, **kwargs
    ):
        super().__init__(name, shape, weight_init, None, resist_scale, p_conn, batch_size=batch_size, **kwargs)

        ## Synapse and BCM hyper-parameters
        self.shape = shape ## shape of synaptic efficacy matrix
        self.tau_w = tau_w ## time constant governing synaptic plasticity
        self.tau_theta = tau_theta ## time constant of threshold delta variables
        self.w_decay = w_decay ## synaptic decay factor
        self.w_bound = w_bound  ## soft weight constraint
        self.Rscale = resist_scale ## post-transformation scale factor
        self.theta0 = theta0 #-1. ## initial condition for theta/threshold variables

        ## Compartment setup
        preVals = jnp.zeros((self.batch_size, shape[0]))
        postVals = jnp.zeros((self.batch_size, shape[1]))
        self.pre = Compartment(preVals) ## pre-synaptic statistic
        self.post = Compartment(postVals) ## post-synaptic statistic
        self.post_term = Compartment(postVals)
        self.theta = Compartment(postVals + self.theta0) ## synaptic modification thresholds
        self.dWeights = Compartment(self.weights.get() * 0)

    @compilable
    def evolve(self, t, dt): #t, dt, tau_w, tau_theta, w_bound, w_decay, pre, post, theta, weights):
        eps = 1e-7
        post_term = self.post.get() * (self.post.get() - self.theta.get())  # post - theta
        post_term = post_term * (1. / (self.theta.get() + eps))
        dWeights = jnp.matmul(self.pre.get().T, post_term)
        if self.w_bound > 0.:
            dWeights = dWeights * (self.w_bound - jnp.abs(self.weights.get()))
        ## update synaptic efficacies according to a leaky ODE
        dWeights = -self.weights.get() * self.w_decay + dWeights
        _W = self.weights.get() + dWeights * dt / self.tau_w
        ## update synaptic modification threshold as a leaky ODE
        dtheta = jnp.mean(jnp.square(self.post.get()), axis=0, keepdims=True)  ## batch avg
        theta = self.theta.get() + (-self.theta.get() + dtheta) * dt / self.tau_theta

        #self.weights.set(weights)
        self.theta.set(theta)
        self.dWeights.set(dWeights)
        self.post_term.set(post_term)

    @compilable
    def reset(self):
        preVals = jnp.zeros((self.batch_size.get(), self.shape.get()[0]))
        postVals = jnp.zeros((self.batch_size.get(), self.shape.get()[1]))

        if not self.inputs.targeted:
            self.inputs.set(preVals)
        self.outputs.set(postVals)
        self.pre.set(preVals)
        self.post.set(postVals)
        self.dWeights.set(jnp.zeros(self.shape.get()))
        self.post_term.set(postVals)

    # def save(self, directory, **kwargs):
    #     file_name = directory + "/" + self.name + ".npz"
    #     jnp.savez(file_name,
    #               weights=self.weights.value, theta=self.theta.value)
    #
    # def load(self, directory, **kwargs):
    #     file_name = directory + "/" + self.name + ".npz"
    #     data = jnp.load(file_name)
    #     self.weights.set(data['weights'])
    #     self.theta.set(data['theta'])

    @classmethod
    def help(cls): ## component help function
        properties = {
            "synapse_type": "BCMSTDPSynapse - performs an adaptable synaptic "
                            "transformation  of inputs to produce output signals; "
                            "synapses are adjusted via BCM theory"
        }
        compartment_props = {
            "input_compartments":
                {"inputs": "Takes in external input signal values",
                 "key": "JAX PRNG key",
                 "pre": "Pre-synaptic statistic for BCM (z_j)",
                 "post": "Post-synaptic statistic for BCM (z_i)"},
            "parameter_compartments":
                {"weights": "Synapse efficacy/strength parameter values",
                 "biases": "Base-rate/bias parameter values"},
            "output_compartments":
                {"outputs": "Output of synaptic transformation",
                 "theta": "Synaptic modification threshold variable",
                 "dWeights": "Synaptic weight value adjustment matrix produced at time t"},
        }
        hyperparams = {
            "shape": "Shape of synaptic weight value matrix; number inputs x number outputs",
            "batch_size": "Batch size dimension of this component",
            "weight_init": "Initialization conditions for synaptic weight (W) values",
            "resist_scale": "Resistance level scaling factor (applied to output of transformation)",
            "p_conn": "Probability of a connection existing (otherwise, it is masked to zero)",
            "tau_theta": "Time constant for synaptic threshold variable `theta`",
            "tau_w": "Time constant for BCM synaptic adjustment",
            "w_bound": "Soft synaptic bound applied to synapses post-update",
            "w_decay": "Synaptic decay term",
            "eta": "Global learning rate",
            "theta0": "Initial condition for theta/threshold variables"
        }
        info = {cls.__name__: properties,
                "compartments": compartment_props,
                "dynamics": "outputs = [(W * Rscale) * inputs] ;"
                            "tau_w dW_{ij}/dt = z_j * (z_i - theta) - W_{ij} * w_decay;"
                            "tau_theta d(theta_{i})/dt = (-theta_{i} + (z_i)^2)",
                "hyperparameters": hyperparams}
        return info

if __name__ == '__main__':
    from ngcsimlib.context import Context
    with Context("Bar") as bar:
        Wab = BCMSynapse("Wab", (2, 3), 0.0004, 1, 1)
    print(Wab)
