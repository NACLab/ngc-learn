# %%

from ngclearn.components.jaxComponent import JaxComponent
from jax import numpy as jnp, jit
from ngclearn.utils.model_utils import sigmoid, d_sigmoid

from ngclearn import compilable #from ngcsimlib.parser import compilable
from ngclearn import Compartment #from ngcsimlib.compartment import Compartment

class BernoulliErrorCell(JaxComponent): ## Rate-coded/real-valued error unit/cell
    """
    A simple (non-spiking) Bernoulli error cell - this is a fixed-point solution
    of a mismatch signal. Specifically, this cell operates as a factorized multivariate
    Bernoulli distribution.

    | --- Cell Input Compartments: ---
    | p - predicted probability (or logits) of positive trial (takes in external signals)
    | target - desired/goal value (takes in external signals)
    | modulator - modulation signal (takes in optional external signals)
    | mask - binary/gating mask to apply to error neuron calculations
    | --- Cell Output Compartments: ---
    | L - local loss function embodied by this cell
    | dp - derivative of L w.r.t. p (or logits, if p = sigmoid(logits))
    | dtarget - derivative of L w.r.t. target

    Args:
        name: the string name of this cell

        n_units: number of cellular entities (neural population size)

        batch_size: batch size dimension of this cell (Default: 1)

        input_logits: if True, treats compartment `p` as logits and will apply a sigmoidal
            link, i.e., _p = sigmoid(p), to obtain the param p for Bern(X=1; p)

    """
    def __init__(self, name, n_units, batch_size=1, input_logits=False, shape=None, **kwargs):
        super().__init__(name, **kwargs)

        ## Layer Size Setup
        _shape = (batch_size, n_units)  ## default shape is 2D/matrix
        if shape is None:
            shape = (n_units,)  ## we set shape to be equal to n_units if nothing provided
        else:
            _shape = (batch_size, shape[0], shape[1], shape[2])  ## shape is 4D tensor
        self.shape = shape
        self.n_units = n_units
        self.batch_size = batch_size
        self.input_logits = input_logits

        ## Convolution shape setup
        self.width = self.height = n_units

        ## Compartment setup
        restVals = jnp.zeros(_shape)
        self.L = Compartment(0., display_name="Bernoulli Log likelihood", units="nats") # loss compartment
        self.p = Compartment(restVals, display_name="Bernoulli param (prob or logit) for B(X=1; p)") # pos trial prob name. input wire
        self.dp = Compartment(restVals) # derivative of positive trial prob
        self.target = Compartment(restVals, display_name="Bernoulli data/target variable") # target. input wire
        self.dtarget = Compartment(restVals) # derivative target
        self.modulator = Compartment(restVals + 1.0) # to be set/consumed
        self.mask = Compartment(restVals + 1.0)

    # @transition(output_compartments=["dp", "dtarget", "L", "mask"])
    @compilable
    def advance_state(self, dt): ## compute Bernoulli error cell output
        # Get the variables
        p = self.p.get()
        target = self.target.get()
        modulator = self.modulator.get()
        mask = self.mask.get()

        # Moves Bernoulli error cell dynamics one step forward. Specifically, this routine emulates the error unit
        # behavior of the local cost functional
        eps = 0.0001
        _p = p
        if self.input_logits: ## convert from "logits" to probs via sigmoidal link function
            _p = sigmoid(p)
        _p = jnp.clip(_p, eps, 1. - eps) ## post-process to prevent div by 0
        x = target
        #sum_x = jnp.sum(x) ## Sum^N_{n=1} x_n (n is n-th datapoint)
        #sum_1mx = jnp.sum(1. - x) ## Sum^N_{n=1} (1 - x_n)

        one_min_p = 1. - _p
        one_min_x = 1. - x
        log_p = jnp.log(_p) ## ln(p)
        log_one_min_p = jnp.log(one_min_p) ## ln(1 - p)
        L = jnp.sum(log_p * x + log_one_min_p * one_min_x) ## Bern LL
        if self.input_logits:
            dL_dp = x - _p ## d(Bern LL)/dp where _p = sigmoid(p)
        else:
            dL_dp = x/(_p) - one_min_x/one_min_p  ## d(Bern LL)/dp
            dL_dp = dL_dp  * d_sigmoid(p)
        dL_dx = (log_p - log_one_min_p)  ## d(Bern LL)/dx
        dp = dL_dp

        dp = dp * modulator * mask ## NOTE: how does mask apply to a multivariate Bernoulli?
        dtarget = dL_dx * modulator * mask
        mask = mask * 0. + 1. ## "eat" the mask as it should only apply at time t

        # Set state
        # dp, dtarget, jnp.squeeze(L), mask
        self.dp.set(dp)
        self.dtarget.set(dtarget)
        self.L.set(jnp.squeeze(L))
        self.mask.set(mask)


    # @transition(output_compartments=["dp", "dtarget", "target", "p", "modulator", "L", "mask"])
    @compilable
    def reset(self): ## reset core components/statistics
        _shape = (self.batch_size, self.shape[0])
        if len(self.shape) > 1:
            _shape = (self.batch_size, self.shape[0], self.shape[1], self.shape[2])
        restVals = jnp.zeros(_shape) ## "rest"/reset values
        dp = restVals
        dtarget = restVals
        target = restVals
        p = restVals
        modulator = restVals + 1. ## reset modulator signal
        L = 0. #jnp.zeros((1, 1)) ## rest loss
        mask = jnp.ones(_shape) ## reset mask

        # Set compartment
        self.dp.set(dp)
        self.dtarget.set(dtarget)
        self.target.set(target)
        self.p.set(p)
        self.modulator.set(modulator)
        self.L.set(L)
        self.mask.set(mask)


    @classmethod
    def help(cls): ## component help function
        properties = {
            "cell_type": "GaussianErrorcell - computes mismatch/error signals at "
                         "each time step t (between a `target` and a prediction `mu`)"
        }
        compartment_props = {
            "inputs":
                {"p": "External input positive probability value(s)",
                 "target": "External input target signal value(s)",
                 "modulator": "External input modulatory/scaling signal(s)",
                 "mask": "External binary/gating mask to apply to signals"},
            "outputs":
                {"L": "Local loss value computed/embodied by this error-cell",
                 "dp": "first derivative of loss w.r.t. positive probability value(s)",
                 "dtarget": "first derivative of loss w.r.t. target value(s)"},
        }
        hyperparams = {
            "n_units": "Number of neuronal cells to model in this layer",
            "batch_size": "Batch size dimension of this component",
            "sigma": "External input variance value (currently fixed and not learnable)"
        }
        info = {cls.__name__: properties,
                "compartments": compartment_props,
                "dynamics": "Bernoulli(x=target; p) where target is binary variable",
                "hyperparameters": hyperparams}
        return info

if __name__ == '__main__':
    from ngcsimlib.context import Context
    with Context("Bar") as bar:
        X = BernoulliErrorCell("X", 9)
    print(X)
