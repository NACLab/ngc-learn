# %%

from ngclearn.components.jaxComponent import JaxComponent
from jax import numpy as jnp, jit
from ngclearn import compilable #from ngcsimlib.parser import compilable
from ngclearn import Compartment #from ngcsimlib.compartment import Compartment

class LaplacianErrorCell(JaxComponent): ## Rate-coded/real-valued error unit/cell
    """
    A simple (non-spiking) Laplacian error cell - this is a fixed-point solution
    of a mismatch/error signal.

    | --- Cell Input Compartments: ---
    | shift - predicted shift value (takes in external signals)
    | Scale - predicted scale (takes in external signals)
    | target - desired/goal value (takes in external signals)
    | modulator - modulation signal (takes in optional external signals)
    | mask - binary/gating mask to apply to error neuron calculations
    | --- Cell Output Compartments: ---
    | L - local loss function embodied by this cell
    | dshift - derivative of L w.r.t. shift
    | dScale - derivative of L w.r.t. Scale
    | dtarget - derivative of L w.r.t. target

    Args:
        name: the string name of this cell

        n_units: number of cellular entities (neural population size)

        batch_size: batch size dimension of this cell (Default: 1)

        scale: initial/fixed value for prediction scale matrix in multivariate laplacian distribution;
            Note that if the compartment `Scale` is never used, then this cell assumes that the scale collapses
            to a constant/fixed `scale`
    """

    def __init__(self, name, n_units, batch_size=1, scale=1., shape=None, **kwargs):
        super().__init__(name, **kwargs)

        ## Layer Size Setup
        _shape = (batch_size, n_units)  ## default shape is 2D/matrix
        if shape is None:
            shape = (n_units,)  ## we set shape to be equal to n_units if nothing provided
        else:
            _shape = (batch_size, shape[0], shape[1], shape[2])  ## shape is 4D tensor
        scale_shape = (1, 1)
        if not isinstance(scale, float) and not isinstance(scale, int):
            scale_shape = jnp.array(scale).shape
        self.scale_shape = scale_shape
        ## Layer Size setup
        self.n_units = n_units
        self.batch_size = batch_size

        ## Convolution shape setup
        self.width = self.height = n_units

        ## Compartment setup
        restVals = jnp.zeros(_shape)
        self.L = Compartment(0., display_name="Laplacian Log likelihood", units="nats") ## loss compartment
        self.shift = Compartment(restVals, display_name="Laplacian shift") ## shift/shift name. input wire
        _Scale = jnp.zeros(scale_shape)
        self.Scale = Compartment(_Scale + scale, display_name="Laplacian scale") ## scale/scale name. input wire
        self.dshift = Compartment(restVals) ## derivative shift
        self.dScale = Compartment(_Scale) ## derivative scale
        self.target = Compartment(restVals, display_name="Laplacian data/target variable") ## target. input wire
        self.dtarget = Compartment(restVals) ## derivative target
        self.modulator = Compartment(restVals + 1.0) ## to be set/consumed
        self.mask = Compartment(restVals + 1.0)

    @compilable
    def advance_state(self, dt): ## compute Laplacian error cell output
        # Get the variables
        shift = self.shift.get()
        target = self.target.get()
        Scale = self.Scale.get()
        modulator = self.modulator.get()
        mask = self.mask.get()

        # Moves Laplacian cell dynamics one step forward. Specifically, this routine emulates the error unit
        # behavior of the local cost functional:
        # FIXME: Currently, below does: L(targ, shift) = -||targ - shift||_1/scale
        #        but should support full log likelihood of the multivariate Laplacian with scale matrix of different types
        # TODO: could introduce a variant of LaplacianErrorCell that moves according to an ODE
        #       (using integration time constant dt)
        _dshift = jnp.sign(target - shift)  # e (error unit)
        dshift = _dshift/Scale
        dtarget = -dshift  # reverse of e
        dScale = Scale * 0 + 1.  # no derivative is calculated at this time for the scale
        L = -jnp.sum(jnp.abs(_dshift)) * (1. / Scale) # technically, this is mean absolute error

        dshift = dshift * modulator * mask
        dtarget = dtarget * modulator * mask
        mask = mask * 0. + 1.  ## "eat" the mask as it should only apply at time t

        # Update compartments
        self.dshift.set(dshift)
        self.dtarget.set(dtarget)
        self.dScale.set(dScale)
        self.L.set(jnp.squeeze(L))
        self.mask.set(mask)

    @compilable
    def reset(self): ## reset core components/statistics
        restVals = jnp.zeros((self.batch_size, self.n_units))
        dshift = restVals
        dtarget = restVals
        dScale = jnp.zeros(self.scale_shape)
        target = restVals
        shift = restVals
        modulator = shift + 1.
        L = 0.
        mask = jnp.ones((self.batch_size, self.n_units))

        self.dshift.set(dshift)
        self.dtarget.set(dtarget)
        self.dScale.set(dScale)
        self.target.set(target)
        self.shift.set(shift)
        self.modulator.set(modulator)
        self.L.set(L)
        self.mask.set(mask)

    @classmethod
    def help(cls): ## component help function
        properties = {
            "cell_type": "LaplacianErrorcell - computes mismatch/error signals at "
                         "each time step t (between a `target` and a prediction `shift`)"
        }
        compartment_props = {
            "inputs":
                {"shift": "External input prediction value(s)",
                 "Scale": "External scale prediction value(s)",
                 "target": "External input target signal value(s)",
                 "modulator": "External input modulatory/scaling signal(s)",
                 "mask": "External binary/gating mask to apply to signals"},
            "outputs":
                {"L": "Local loss value computed/embodied by this error-cell",
                 "dshift": "first derivative of loss w.r.t. prediction value(s)",
                 "dScale": "first derivative of loss w.r.t. scale value(s)",
                 "dtarget": "first derivative of loss w.r.t. target value(s)"},
        }
        hyperparams = {
            "n_units": "Number of neurons to model in this layer",
            "batch_size": "Batch size dimension of this component"
        }
        info = {cls.__name__: properties,
                "compartments": compartment_props,
                "dynamics": "Laplacian(x=target; shift, scale)",
                "hyperparameters": hyperparams}
        return info

if __name__ == '__main__':
    from ngcsimlib.context import Context
    with Context("Bar") as bar:
        X = LaplacianErrorCell("X", 9)
    print(X)
