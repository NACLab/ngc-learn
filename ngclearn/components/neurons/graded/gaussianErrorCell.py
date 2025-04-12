from ngclearn import resolver, Component, Compartment
from ngclearn.components.jaxComponent import JaxComponent
from jax import numpy as jnp, jit
from ngclearn.utils import tensorstats
from ngcsimlib.compilers.process import transition

class GaussianErrorCell(JaxComponent): ## Rate-coded/real-valued error unit/cell
    """
    A simple (non-spiking) Gaussian error cell - this is a fixed-point solution
    of a mismatch signal.

    | --- Cell Input Compartments: ---
    | mu - predicted value (takes in external signals)
    | Sigma - predicted covariance (takes in external signals)
    | target - desired/goal value (takes in external signals)
    | modulator - modulation signal (takes in optional external signals)
    | mask - binary/gating mask to apply to error neuron calculations
    | --- Cell Output Compartments: ---
    | L - local loss function embodied by this cell
    | dmu - derivative of L w.r.t. mu
    | dSigma - derivative of L w.r.t. Sigma
    | dtarget - derivative of L w.r.t. target

    Args:
        name: the string name of this cell

        n_units: number of cellular entities (neural population size)

        batch_size: batch size dimension of this cell (Default: 1)

        sigma: initial/fixed value for prediction covariance matrix (𝚺) in multivariate gaussian distribution;
            Note that if the compartment `Sigma` is never used, then this cell assumes that the covariance collapses
            to a constant/fixed `sigma`
    """
    def __init__(self, name, n_units, batch_size=1, sigma=1., shape=None, **kwargs):
        super().__init__(name, **kwargs)

        ## Layer Size Setup
        _shape = (batch_size, n_units)  ## default shape is 2D/matrix
        if shape is None:
            shape = (n_units,)  ## we set shape to be equal to n_units if nothing provided
        else:
            _shape = (batch_size, shape[0], shape[1], shape[2])  ## shape is 4D tensor
        sigma_shape = (1,1)
        if not isinstance(sigma, float) and not isinstance(sigma, int):
            sigma_shape = jnp.array(sigma).shape
        self.sigma_shape = sigma_shape
        self.shape = shape
        self.n_units = n_units
        self.batch_size = batch_size

        ## Convolution shape setup
        self.width = self.height = n_units

        ## Compartment setup
        restVals = jnp.zeros(_shape)
        self.L = Compartment(0., display_name="Gaussian Log likelihood", units="nats") # loss compartment
        self.mu = Compartment(restVals, display_name="Gaussian mean") # mean/mean name. input wire
        self.dmu = Compartment(restVals) # derivative mean
        _Sigma = jnp.zeros(sigma_shape)
        self.Sigma = Compartment(_Sigma + sigma, display_name="Gaussian variance/covariance")
        self.dSigma = Compartment(_Sigma)
        self.target = Compartment(restVals, display_name="Gaussian data/target variable") # target. input wire
        self.dtarget = Compartment(restVals) # derivative target
        self.modulator = Compartment(restVals + 1.0) # to be set/consumed
        self.mask = Compartment(restVals + 1.0)

    @transition(output_compartments=["dmu", "dtarget", "dSigma", "L", "mask"])
    @staticmethod
    def advance_state(dt, mu, target, Sigma, modulator, mask): ## compute Gaussian error cell output
        # Moves Gaussian cell dynamics one step forward. Specifically, this routine emulates the error unit
        # behavior of the local cost functional:
        # FIXME: Currently, below does: L(targ, mu) = -(1/(2*sigma)) * ||targ - mu||^2_2
        #        but should support full log likelihood of the multivariate Gaussian with covariance of different types
        # TODO: could introduce a variant of GaussianErrorCell that moves according to an ODE
        #       (using integration time constant dt)
        _dmu = (target - mu)  # e (error unit)
        dmu = _dmu / Sigma
        dtarget = -dmu  # reverse of e
        dSigma = Sigma * 0 + 1. # no derivative is calculated at this time for sigma
        L = -jnp.sum(jnp.square(_dmu)) * (0.5 / Sigma)

        dmu = dmu * modulator * mask ## not sure how mask will apply to a full covariance...
        dtarget = dtarget * modulator * mask
        mask = mask * 0. + 1. ## "eat" the mask as it should only apply at time t
        return dmu, dtarget, dSigma, jnp.squeeze(L), mask

    @transition(output_compartments=["dmu", "dtarget", "dSigma", "target", "mu", "modulator", "L", "mask"])
    @staticmethod
    def reset(batch_size, shape, sigma_shape): ## reset core components/statistics
        _shape = (batch_size, shape[0])
        if len(shape) > 1:
            _shape = (batch_size, shape[0], shape[1], shape[2])
        restVals = jnp.zeros(_shape)
        dmu = restVals
        dtarget = restVals
        dSigma = jnp.zeros(sigma_shape)
        target = restVals
        mu = restVals
        modulator = mu + 1.
        L = 0. #jnp.zeros((1, 1))
        mask = jnp.ones(_shape)
        return dmu, dtarget, dSigma, target, mu, modulator, L, mask

    @classmethod
    def help(cls): ## component help function
        properties = {
            "cell_type": "GaussianErrorcell - computes mismatch/error signals at "
                         "each time step t (between a `target` and a prediction `mu`)"
        }
        compartment_props = {
            "inputs":
                {"mu": "External input prediction value(s)",
                 "Sigma": "External variance/covariance prediction value(s)",
                 "target": "External input target signal value(s)",
                 "modulator": "External input modulatory/scaling signal(s)",
                 "mask": "External binary/gating mask to apply to signals"},
            "outputs":
                {"L": "Local loss value computed/embodied by this error-cell",
                 "dmu": "first derivative of loss w.r.t. prediction value(s)",
                 "dSigma": "first derivative of loss w.r.t. variance/covariance value(s)",
                 "dtarget": "first derivative of loss w.r.t. target value(s)"},
        }
        hyperparams = {
            "n_units": "Number of neuronal cells to model in this layer",
            "batch_size": "Batch size dimension of this component",
            "sigma": "External input variance value (currently fixed and not learnable)"
        }
        info = {cls.__name__: properties,
                "compartments": compartment_props,
                "dynamics": "Gaussian(x=target; mu, sigma)",
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
        X = GaussianErrorCell("X", 9)
    print(X)
