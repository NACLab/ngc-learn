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
    return _run_laplacian_cell(dt, targ, mu)

@jit
def _run_laplacian_cell(dt, targ, mu):
    """
    Moves Laplacian cell dynamics one step forward. Specifically, this
    routine emulates the error unit behavior of the local cost functional:

    | L(targ, mu) = -||targ - mu||_1
    | or log likelihood of the Laplace distribution with identity scale

    Args:
        dt: integration time constant

        targ: target pattern value

        mu: prediction value

    Returns:
        derivative w.r.t. mean "dmu", derivative w.r.t. target dtarg, loss
    """
    dmu = jnp.sign(targ - mu) # e (error unit)
    dtarg = -dmu # reverse of e
    L = -jnp.sum(jnp.abs(dmu)) # technically, this is mean absolute error
    return dmu, dtarg, L

class LaplacianErrorCell(JaxComponent): ## Rate-coded/real-valued error unit/cell
    """
    A simple (non-spiking) Laplacian error cell - this is a fixed-point solution
    of a mismatch/error signal.

    | --- Cell Input Compartments: ---
    | mu - predicted value (takes in external signals)
    | target - desired/goal value (takes in external signals)
    | modulator - modulation signal (takes in optional external signals)
    | mask - binary/gating mask to apply to error neuron calculations
    | --- Cell Output Compartments: ---
    | L - local loss function embodied by this cell
    | dmu - derivative of L w.r.t. mu
    | dtarget - derivative of L w.r.t. target

    Args:
        name: the string name of this cell

        n_units: number of cellular entities (neural population size)

        tau_m: (Unused -- currently cell is a fixed-point model)

        leakRate: (Unused -- currently cell is a fixed-point model)
    """

    # Define Functions
    def __init__(self, name, n_units, batch_size=1, **kwargs):
        super().__init__(name, **kwargs)

        ## Layer Size setup
        self.n_units = n_units
        self.batch_size = batch_size

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
        self.mask = Compartment(restVals + 1.0)

    @staticmethod
    def _advance_state(dt, mu, target, modulator, mask):
        ## compute Laplacian error cell output
        dmu, dtarget, L = _run_cell(dt, target * mask, mu * mask)
        dmu = dmu * modulator * mask
        dtarget = dtarget * modulator * mask
        mask = mask * 0. + 1.  ## "eat" the mask as it should only apply at time t
        return dmu, dtarget, L, mask

    @resolver(_advance_state)
    def advance_state(self, dmu, dtarget, L, mask):
        self.dmu.set(dmu)
        self.dtarget.set(dtarget)
        self.L.set(L)
        self.mask.set(mask)

    @staticmethod
    def _reset(batch_size, n_units):
        restVals = jnp.zeros((batch_size, n_units))
        dmu = restVals
        dtarget = restVals
        target = restVals
        mu = restVals
        modulator = mu + 1.
        L = 0.
        mask = jnp.ones((batch_size, n_units))
        return dmu, dtarget, target, mu, modulator, L, mask

    @resolver(_reset)
    def reset(self, dmu, dtarget, target, mu, modulator, L, mask):
        self.dmu.set(dmu)
        self.dtarget.set(dtarget)
        self.target.set(target)
        self.mu.set(mu)
        self.modulator.set(modulator)
        self.L.set(L)
        self.mask.set(mask)

    @classmethod
    def help(cls): ## component help function
        properties = {
            "cell_type": "LaplacianErrorcell - computes mismatch/error signals at "
                         "each time step t (between a `target` and a prediction `mu`)"
        }
        compartment_props = {
            "inputs":
                {"mu": "External input prediction value(s)",
                 "target": "External input target signal value(s)",
                 "modulator": "External input modulatory/scaling signal(s)",
                 "mask": "External binary/gating mask to apply to signals"},
            "outputs":
                {"L": "Local loss value computed/embodied by this error-cell",
                 "dmu": "first derivative of loss w.r.t. prediction value(s)",
                 "dtarget": "first derivative of loss w.r.t. target value(s)"},
        }
        hyperparams = {
            "n_units": "Number of neurons to model in this layer",
            "batch_size": "Batch size dimension of this component"
        }
        info = {cls.__name__: properties,
                "compartments": compartment_props,
                "dynamics": "Laplacian(x=target; shift=mu, scale=1)",
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
        X = LaplacianErrorCell("X", 9)
    print(X)
