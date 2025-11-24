import numpy as np
from ngclearn.utils.distribution_generator import DistributionGenerator as dist
from ngclearn import numpy as jnp

from jax import numpy as jnp, random, jit
from ngclearn import Context, MethodProcess
from ngclearn.components.synapses.hebbian.hebbianSynapse import HebbianSynapse
from ngclearn.components.neurons.graded.gaussianErrorCell import GaussianErrorCell
from ngcsimlib.global_state import stateManager

class Iterative_Ridge():
    """
        A neural circuit implementation of the iterative Ridge (L2) algorithm
        using a Hebbian learning update rule.

        This circuit implements sparse regression through Hebbian synapses with L2 regularization.

        The specific differential equation that characterizes this model is adding lmbda * W
        to the dW (the gradient of loss/energy function):
        | dW/dt = dW + lmbda * W

        

        | --- Circuit Components: ---
        | W - HebbianSynapse for learning regularized dictionary weights
        | err - GaussianErrorCell for computing prediction errors
        | --- Component Compartments ---
        | W.inputs - input features (takes in external signals)
        | W.pre - pre-synaptic activity for Hebbian learning
        | W.post - post-synaptic error signals
        | W.weights - learned dictionary coefficients
        | err.mu - predicted outputs
        | err.target - target signals (target vector)
        | err.dmu - error gradients
        | err.L - loss/energy values

        Args:
            key: JAX PRNG key for random number generation

            name: string name for this solver

            sys_dim: dimensionality of the system/target space

            dict_dim: dimensionality of the dictionary/feature space/the number of predictors

            batch_size: number of samples to process in parallel

            weight_fill: initial constant value to fill weight matrix with (Default: 0.05)

            lr: learning rate for synaptic weight updates (Default: 0.01)

            ridge_lmbda: L2 regularization lambda parameter (Default: 0.0001)

            optim_type: optimization type for updating weights; supported values are
                "sgd" and "adam" (Default: "adam")

            threshold: minimum absolute coefficient value - values below this are set
                to zero during thresholding (Default: 0.001)

            epochs: number of training epochs (Default: 100)
    """
    def __init__(self, key, name, sys_dim, dict_dim, batch_size, weight_fill=0.05, lr=0.01,
                 ridge_lmbda=0.0001, optim_type="adam", threshold=0.001, epochs=100):
        key, *subkeys = random.split(key, 10)

        self.T = 100
        self.dt = 1
        self.epochs = epochs
        self.weight_fill = weight_fill
        self.threshold = threshold
        self.name = name
        self.lr = lr
        feature_dim = dict_dim

        with Context(self.name) as self.circuit:
            self.W = HebbianSynapse(
                "W", shape=(feature_dim, sys_dim), eta=self.lr, sign_value=-1, 
                weight_init=dist.constant(value=weight_fill), prior=('ridge', ridge_lmbda), w_bound=0.,
                optim_type=optim_type, key=subkeys[0]
            )
            self.err = GaussianErrorCell("err", n_units=sys_dim)

            # # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            self.W.batch_size = batch_size
            self.err.batch_size = batch_size
            # # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            self.W.outputs >> self.err.mu
            self.err.dmu >> self.W.post
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

            advance = (MethodProcess(name="advance_state")
                               >> self.W.advance_state
                               >> self.err.advance_state)
            self.advance = advance

            evolve = (MethodProcess(name="evolve")
                      >> self.W.evolve)
            self.evolve = evolve

            reset = (MethodProcess(name="reset")
                     >> self.err.reset
                     >> self.W.reset)
            self.reset = reset
            
    def batch_set(self, batch_size):
        self.W.batch_size = batch_size
        self.err.batch_size = batch_size

    def clamp(self, y_scaled, X):
        self.W.inputs.set(X)
        self.W.pre.set(X)
        self.err.target.set(y_scaled)

    def thresholding(self, scale=2):
        coef_old = self.coef_ #self.W.weights.value
        new_coeff = jnp.where(jnp.abs(coef_old) >= self.threshold, coef_old, 0.)

        self.coef_ = new_coeff * scale
        self.W.weights.set(new_coeff)

        return self.coef_, coef_old


    def fit(self, y, X):
        self.reset.run()
        self.clamp(y_scaled=y, X=X)

        for i in range(self.epochs):
            inputs = jnp.array(self.advance.pack_rows(self.T, t=lambda x: x, dt=self.dt))
            stateManager.state, outputs = self.advance.scan(inputs)
            self.evolve.run(t=self.T, dt=self.dt)

        self.coef_ = np.array(self.W.weights.get())

        return self.coef_, self.err.mu.get(), self.err.L.get()

