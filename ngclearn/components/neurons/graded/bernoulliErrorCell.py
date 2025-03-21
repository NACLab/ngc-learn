from ngclearn import resolver, Component, Compartment
from ngclearn.components.jaxComponent import JaxComponent
from jax import numpy as jnp, jit
from ngclearn.utils import tensorstats

class BernoulliErrorCell(JaxComponent): ## Rate-coded/real-valued error unit/cell
    """
    A simple (non-spiking) Bernoulli error cell - this is a fixed-point solution
    of a mismatch signal. Specifically, this cell operates as a factorized multivariate 
    Bernoulli distribution.

    | --- Cell Input Compartments: ---
    | p - predicted probability of positive trial (takes in external signals)
    | target - desired/goal value (takes in external signals)
    | modulator - modulation signal (takes in optional external signals)
    | mask - binary/gating mask to apply to error neuron calculations
    | --- Cell Output Compartments: ---
    | L - local loss function embodied by this cell
    | dp - derivative of L w.r.t. p
    | dtarget - derivative of L w.r.t. target

    Args:
        name: the string name of this cell

        n_units: number of cellular entities (neural population size)

        batch_size: batch size dimension of this cell (Default: 1)

    """
    def __init__(self, name, n_units, batch_size=1, shape=None, **kwargs):
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

        ## Convolution shape setup
        self.width = self.height = n_units

        ## Compartment setup
        restVals = jnp.zeros(_shape)
        self.L = Compartment(0., display_name="Bernoulli Log likelihood", units="nats") # loss compartment
        self.p = Compartment(restVals, display_name="Bernoulli prob for B(X=1; p)") # pos trial prob name. input wire
        self.dp = Compartment(restVals) # derivative of positive trial prob
        self.target = Compartment(restVals, display_name="Bernoulli data/target variable") # target. input wire
        self.dtarget = Compartment(restVals) # derivative target
        self.modulator = Compartment(restVals + 1.0) # to be set/consumed
        self.mask = Compartment(restVals + 1.0)

    @staticmethod
    def _advance_state(dt, p, target, modulator, mask): ## compute Bernoulli error cell output
        # Moves Bernoulli error cell dynamics one step forward. Specifically, this routine emulates the error unit
        # behavior of the local cost functional
        eps = 0.001
        _p = jnp.clip(p, eps, 1. - eps) ## to prevent division by 0 later on
        x = target
        sum_x = jnp.sum(x) ## Sum^N_{n=1} x_n (n is n-th datapoint)
        sum_1mx = jnp.sum(1. - x) ## Sum^N_{n=1} (1 - x_n)
        log_p = jnp.log(_p) ## log(p)
        log_1mp = jnp.log(1. - _p) ## log(1 - p)
        L = log_p * sum_x + log_1mp * sum_1mx ## Bern LL
        dL_dp = sum_x/log_p - sum_1mx/log_1mp ## d(Bern LL)/dp 
        dL_dx = log_p - log_1mp ## d(Bern LL)/dx

        dp = dL_dp * modulator * mask ## not sure how mask will apply to a full covariance...
        dtarget = dL_dx * modulator * mask
        mask = mask * 0. + 1. ## "eat" the mask as it should only apply at time t
        return dp, dtarget, jnp.squeeze(L), mask

    @resolver(_advance_state)
    def advance_state(self, dp, dtarget, L, mask):
        self.dp.set(dp)
        self.dtarget.set(dtarget)
        self.L.set(L)
        self.mask.set(mask)

    @staticmethod
    def _reset(batch_size, shape): ## reset core components/statistics
        _shape = (batch_size, shape[0])
        if len(shape) > 1:
            _shape = (batch_size, shape[0], shape[1], shape[2])
        restVals = jnp.zeros(_shape)
        dp = restVals
        dtarget = restVals
        target = restVals
        p = restVals
        modulator = mu + 1.
        L = 0. #jnp.zeros((1, 1))
        mask = jnp.ones(_shape)
        return dp, dtarget, target, p, modulator, L, mask

    @resolver(_reset)
    def reset(self, dp, dtarget, target, p, modulator, L, mask):
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

if __name__ == '__main__':
    from ngcsimlib.context import Context
    with Context("Bar") as bar:
        X = BernoulliErrorCell("X", 9)
    print(X)
