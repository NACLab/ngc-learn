from ngclearn import resolver, Component, Compartment
from ngclearn.components.jaxComponent import JaxComponent
from jax import numpy as jnp, jit
from ngclearn.utils import tensorstats

def _run_cell(dt, targ, mu):
    """
    Moves cell dynamics one step forward.

    Args:
        dt: integration time constant

        targ: target pattern value

        mu: prediction value

    Returns:
        derivative w.r.t. mean "dmu", derivative w.r.t. target dtarg, local loss
    """
    return _run_gaussian_cell(dt, targ, mu)

@jit
def _run_gaussian_cell(dt, targ, mu):
    """
    Moves Gaussian cell dynamics one step forward. Specifically, this
    routine emulates the error unit behavior of the local cost functional:

    | L(targ, mu) = -(1/2) * ||targ - mu||^2_2
    | or log likelihood of the multivariate Gaussian with identity covariance

    Args:
        dt: integration time constant

        targ: target pattern value

        mu: prediction value

    Returns:
        derivative w.r.t. mean "dmu", derivative w.r.t. target dtarg, loss
    """
    dmu = (targ - mu) # e (error unit)
    dtarg = -dmu # reverse of e
    L = -jnp.sum(jnp.square(dmu)) * 0.5
    return dmu, dtarg, L

class GaussianErrorCell(JaxComponent): ## Rate-coded/real-valued error unit/cell
    """
    A simple (non-spiking) Gaussian error cell - this is a fixed-point solution
    of a mismatch signal.

    | --- Cell Input Compartments: ---
    | mu - predicted value (takes in external signals)
    | target - desired/goal value (takes in external signals)
    | modulator - modulation signal (takes in optional external signals)
    | --- Cell Output Compartments: ---
    | L - local loss function embodied by this cell
    | dmu - derivative of L w.r.t. mu
    | dtarget - derivative of L w.r.t. target

    Args:
        name: the string name of this cell

        n_units: number of cellular entities (neural population size)

        refract_time: relative refractory period time (ms; Default: 0 ms)
    """
    def __init__(self, name, n_units, refract_time=0., batch_size=1, **kwargs):
        super().__init__(name, **kwargs)

        ## Layer Size Setup
        self.n_units = n_units
        self.batch_size = batch_size
        self.refract_T = refract_time  # ms ## refractory period

        ## Convolution shape setup
        self.width = self.height = n_units

        ## Compartment setup
        restVals = jnp.zeros((self.batch_size, self.n_units))
        self.L = Compartment(0.) # loss compartment
        self.mu = Compartment(restVals) # mean/mean name. input wire
        self.dmu = Compartment(restVals) # derivative mean
        self.target = Compartment(restVals) # target. input wire
        self.dtarget = Compartment(restVals) # derivative target
        self.modulator = Compartment(restVals + 1.0) # to be set/consumed
        self.rfr = Compartment(restVals + self.refract_T) ## refractory variable(s)

    @staticmethod
    def _advance_state(dt, refract_T, mu, dmu, target, dtarget, modulator, rfr):
        mask = (rfr >= refract_T) * 1.
        ## compute Gaussian error cell output
        dmu, dtarget, L = _run_cell(dt, target * mask, mu * mask)
        dmu = dmu * modulator * mask
        dtarget = dtarget * modulator * mask
        if refract_T > 0.: ## if non-zero refractory times used, then...
            rfr = (rfr + dt) * (1. - target) + target * dt  # set refract to dt
        return dmu, dtarget, L, rfr

    @resolver(_advance_state)
    def advance_state(self, dmu, dtarget, L, rfr):
        self.dmu.set(dmu)
        self.dtarget.set(dtarget)
        self.L.set(L)
        self.rfr.set(rfr)

    @staticmethod
    def _reset(refract_T, batch_size, n_units):
        restVals = jnp.zeros((batch_size, n_units))
        dmu = restVals
        dtarget = restVals
        target = restVals
        mu = restVals
        modulator = mu + 1.
        L = 0.
        rfr = restVals + refract_T
        return dmu, dtarget, target, mu, modulator, L, rfr

    @resolver(_reset)
    def reset(self, dmu, dtarget, target, mu, modulator, L, rfr):
        self.dmu.set(dmu)
        self.dtarget.set(dtarget)
        self.target.set(target)
        self.mu.set(mu)
        self.modulator.set(modulator)
        self.L.set(L)
        self.rfr.set(rfr)

    @classmethod
    def help(cls): ## component help function
        properties = {
            "cell_type": "GaussianErrorcell - computes mismatch/error signals at "
                         "each time step t (between a `target` and a prediction `mu`)"
        }
        compartment_props = {
            "inputs":
                {"mu": "External input prediction value(s)",
                 "target": "External input target signal value(s)",
                 "modulator": "External input modulatory/scaling signal(s)"},
            "outputs":
                {"L": "Local loss value computed/embodied by this error-cell",
                 "dmu": "first derivative of loss w.r.t. prediction value(s)",
                 "dtarget": "first derivative of loss w.r.t. target value(s)"},
        }
        hyperparams = {
            "n_units": "Number of neuronal cells to model in this layer",
            "batch_size": "Batch size dimension of this component"
        }
        info = {cls.__name__: properties,
                "compartments": compartment_props,
                "dynamics": "Gaussian(x=target; mu, sigma=1)",
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
