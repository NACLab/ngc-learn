from ngclib.component import Component
from jax import numpy as jnp, random, jit
from functools import partial
import time, sys

#@partial(jit, static_argnums=[3])
def run_cell(dt, targ, mu, eType="gaussian"):
    dmu = None
    dtarg = None
    if eType == "gaussian":
        return run_gaussian_cell(dt, targ, mu)
    return dmu, dtarg

@jit
def run_gaussian_cell(dt, targ, mu):
    dmu = (targ - mu) # e (error unit)
    dtarg = -dmu # reverse of e
    return dmu, dtarg

class ErrorCell(Component): ## Rate-coded/real-valued error unit/cell
    ## Class Methods for Compartment Names
    @classmethod
    def inputCompartmentName(cls):
        return 'j' ## electrical current

    @classmethod
    def outputCompartmentName(cls):
        return 'e' ## rate-coded output

    @classmethod
    def meanName(cls):
        return 'mu'

    @classmethod
    def derivMeanName(cls):
        return 'dmu'

    @classmethod
    def targetName(cls):
        return 'target'

    @classmethod
    def derivTargetName(cls):
        return 'dtarget'

    @classmethod
    def modulatorName(cls):
        return 'modulator'

    ## Bind Properties to Compartments for ease of use
    @property
    def mean(self):
        return self.compartments.get(self.meanName(), None)

    @mean.setter
    def mean(self, inp):
        if inp is not None:
            if inp.shape[1] != self.n_units:
                raise RuntimeError("Mean compartment size does not match provided input size " + str(inp.shape) + "for "
                                   + str(self.name))
        self.compartments[self.meanName()] = inp

    @property
    def derivMean(self):
        return self.compartments.get(self.derivMeanName(), None)

    @derivMean.setter
    def derivMean(self, inp):
        if inp is not None:
            if inp.shape[1] != self.n_units:
                raise RuntimeError("Mean.derivative compartment size does not match provided input size " + str(inp.shape) + "for "
                                   + str(self.name))
        self.compartments[self.derivMeanName()] = inp

    @property
    def target(self):
        return self.compartments.get(self.targetName(), None)

    @target.setter
    def target(self, inp):
        if inp is not None:
            if inp.shape[1] != self.n_units:
                raise RuntimeError("Target compartment size does not match provided input size " + str(inp.shape) + "for "
                                   + str(self.name))
        self.compartments[self.targetName()] = inp

    @property
    def derivTarget(self):
        return self.compartments.get(self.derivTargetName(), None)

    @derivTarget.setter
    def derivTarget(self, inp):
        if inp is not None:
            if inp.shape[1] != self.n_units:
                raise RuntimeError("Target.derivative compartment size does not match provided input size " + str(inp.shape) + "for "
                                   + str(self.name))
        self.compartments[self.derivTargetName()] = inp

    @property
    def modulator(self):
        return self.compartments.get(self.modulatorName(), None)

    @modulator.setter
    def modulator(self, inp):
        if inp is not None:
            if inp.shape[1] != self.n_units:
                raise RuntimeError("Modulator compartment size does not match provided input size " + str(inp.shape) + "for "
                                   + str(self.name))
        self.compartments[self.modulatorName()] = inp

    # Define Functions
    def __init__(self, name, n_units, tau_m=0., leakRate=0., key=None, useVerboseDict=False,
                 directory=None, **kwargs):
        super().__init__(name, useVerboseDict, **kwargs)

        ##Random Number Set up
        self.key = key
        if self.key is None:
            self.key = random.PRNGKey(time.time_ns())

        ##Layer Size Setup
        self.n_units = n_units
        self.batch_size = 1

        ## Set up bundle for multiple inputs of current
        self.reset()

    def verify_connections(self):
        #self.metadata.check_incoming_connections(self.inputCompartmentName(), min_connections=1)
        self.metadata.check_incoming_connections(self.meanName(), min_connections=1)
        self.metadata.check_incoming_connections(self.targetName(), min_connections=1)

    def advance_state(self, t, dt, **kwargs):
        ## currently only Gaussian error cells supported
        self.derivMean, self.derivTarget = run_cell(dt, self.target, self.mean, eType="gaussian")
        if self.modulator is not None:
            self.derivMean = self.derivMean * self.modulator
            self.derivTarget = self.derivTarget * self.modulator
            self.modulator = None ## use and consume modulator

    def reset(self, **kwargs):
        self.derivMean = jnp.zeros((self.batch_size, self.n_units))
        self.derivTarget = jnp.zeros((self.batch_size, self.n_units))
        self.target = jnp.zeros((self.batch_size, self.n_units)) #None
        self.mean = jnp.zeros((self.batch_size, self.n_units)) #None
        self.modulator = None

    def save(self, **kwargs):
        pass
