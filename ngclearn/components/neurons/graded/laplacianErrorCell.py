from ngclearn import resolver, Component, Compartment
from ngclearn.components.jaxComponent import JaxComponent
from jax import numpy as jnp, jit
from ngclearn.utils import tensorstats

def run_cell(dt, targ, mu):
    """
    Moves cell dynamics one step forward.

    Args:
        dt: integration time constant

        targ: target pattern value

        mu: prediction value

    Returns:
        derivative w.r.t. mean "dmu", derivative w.r.t. target dtarg, local loss
    """
    return run_laplacian_cell(dt, targ, mu)

@jit
def run_laplacian_cell(dt, targ, mu):
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

    | --- Cell Compartments: ---
    | mu - predicted value (takes in external signals)
    | target - desired/goal value (takes in external signals)
    | L - local loss function embodied by this cell
    | dmu - derivative of L w.r.t. mu
    | dtarget - derivative of L w.r.t. target
    | modulator - modulation signal (takes in optional external signals)

    Args:
        name: the string name of this cell

        n_units: number of cellular entities (neural population size)

        tau_m: (Unused -- currently cell is a fixed-point model)

        leakRate: (Unused -- currently cell is a fixed-point model)
    """

    # Define Functions
    def __init__(self, name, n_units, **kwargs):
        super().__init__(name, **kwargs)

        ## Layer Size setup
        self.n_units = n_units
        self.batch_size = 1

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

    @staticmethod
    def _advance_state(t, dt, mu, dmu, target, dtarget, modulator):
        ## compute Laplacian error cell output
        dmu, dtarget, L = run_cell(dt, target, mu)
        dmu = dmu * modulator
        dtarget = dtarget * modulator
        return dmu, dtarget, L

    @resolver(_advance_state)
    def advance_state(self, dmu, dtarget, L):
        self.dmu.set(dmu)
        self.dtarget.set(dtarget)
        self.L.set(L)

    @staticmethod
    def _reset(batch_size, n_units):
        dmu = jnp.zeros((batch_size, n_units))
        dtarget = jnp.zeros((batch_size, n_units))
        target = jnp.zeros((batch_size, n_units)) #None
        mu = jnp.zeros((batch_size, n_units)) #None
        modulator = mu + 1.
        L = 0.
        return dmu, dtarget, target, mu, modulator, L

    @resolver(_reset)
    def reset(self, dmu, dtarget, target, mu, modulator, L):
        self.dmu.set(dmu)
        self.dtarget.set(dtarget)
        self.target.set(target)
        self.mu.set(mu)
        self.modulator.set(modulator)
        self.L.set(L)

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
