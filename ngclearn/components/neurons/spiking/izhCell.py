from ngclib.component import Component
from jax import numpy as jnp, random, jit, nn
from functools import partial
import time, sys

@jit
def update_times(t, s, tols):
    """
    Updates time-of-last-spike (tols) variable.

    Args:
        t: current time (a scalar/int value)

        s: binary spike vector

        tols: current time-of-last-spike variable

    Returns:
        updated tols variable
    """
    _tols = (1. - s) * tols + (s * t)
    return _tols

class IzhikevichCell(Component): ## Izhikevich neuronal cell
    """
    A spiking cell based on Izhikevich's model of neuronal dynamics.
    (Note that this cell is under construction -- will do nothing at the
    moment.)

    Args:
        name: the string name of this cell

        n_units: number of cellular entities (neural population size)

        tau_m: membrane time constant

        R_m: membrane resistance value

        thr: base value for adaptive thresholds that govern short-term
            plasticity (in milliVolts, or mV)

        v_rest: membrane resting potential (in mV)

        key: PRNG key to control determinism of any underlying random values
            associated with this cell

        useVerboseDict: triggers slower, verbose dictionary mode (Default: False)

        directory: string indicating directory on disk to save Izhikevich parameter
            values to (i.e., initial threshold values and any persistent adaptive
            threshold values)
    """

    ## Class Methods for Compartment Names
    @classmethod
    def inputCompartmentName(cls):
        return 'j' ## electrical current

    @classmethod
    def outputCompartmentName(cls):
        return 's' ## spike/action potential

    @classmethod
    def timeOfLastSpikeCompartmentName(cls):
        return 'tols' ## time-of-last-spike (record vector)

    @classmethod
    def voltageCompartmentName(cls):
        return 'v' ## membrane potential/voltage

    @classmethod
    def refractCompartmentName(cls):
        return 'rfr' ## refractory variable(s)

    ## Bind Properties to Compartments for ease of use
    @property
    def current(self):
        return self.compartments.get(self.inputCompartmentName(), None)

    @current.setter
    def current(self, inp):
        self.compartments[self.inputCompartmentName()] = inp

    @property
    def spikes(self):
        return self.compartments.get(self.outputCompartmentName(), None)

    @spikes.setter
    def spikes(self, out):
        self.compartments[self.outputCompartmentName()] = out

    @property
    def timeOfLastSpike(self):
        return self.compartments.get(self.timeOfLastSpikeCompartmentName(), None)

    @timeOfLastSpike.setter
    def timeOfLastSpike(self, t):
        self.compartments[self.timeOfLastSpikeCompartmentName()] = t

    @property
    def voltage(self):
        return self.compartments.get(self.voltageCompartmentName(), None)

    @voltage.setter
    def voltage(self, v):
        self.compartments[self.voltageCompartmentName()] = v

    @property
    def refract(self):
        return self.compartments.get(self.refractCompartmentName(), None)

    @refract.setter
    def refract(self, rfr):
        self.compartments[self.refractCompartmentName()] = rfr

    @threshold_theta.setter
    def threshold_theta(self, thr):
        self.compartments[self.thresholdThetaName()] = thr

    # Define Functions
    def __init__(self, name, n_units, tau_m, R_m, thr=45., v_rest=-65.,
                 useVerboseDict=False, directory=None, **kwargs):
        super().__init__(name, useVerboseDict, **kwargs)

        ##Random Number Set up
        self.key = key
        if self.key is None:
            self.key = random.PRNGKey(time.time_ns())

        ##Layer Size Setup
        self.n_units = n_units

        self.reset()

    def verify_connections(self):
        pass
        #self.metadata.check_incoming_connections(self.inputCompartmentName(), min_connections=1)

    def advance_state(self, t, dt, **kwargs):
        pass

    def reset(self, **kwargs):
        pass

    def save(self, directory, **kwargs):
        pass
        #file_name = directory + "/" + self.name + ".npz"
        #jnp.savez(file_name, threshold_theta=self.threshold_theta)

    def load(self, directory, **kwargs):
        pass
        #file_name = directory + "/" + self.name + ".npz"
        #data = jnp.load(file_name)
        #self.threshold_theta = data['threshold_theta']
