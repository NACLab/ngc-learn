from ngclib.component import Component
from jax import numpy as jnp, random, jit
from functools import partial
import time

@jit
def update_times(t, s, tols):
    _tols = (1. - s) * tols + (s * t)
    return _tols

@partial(jit, static_argnums=[3])
def sample_poisson(dkey, data, dt, fmax=63.75):
    """
    Samples a Poisson spike train on-the-fly
    """
    pspike = data * (dt/1000.) * fmax
    eps = random.uniform(dkey, data.shape, minval=0., maxval=1., dtype=jnp.float32)
    s_t = (eps < pspike).astype(jnp.float32)
    return s_t

class PoissonCell(Component):
  ## Class Methods for Compartment Names
  @classmethod
  def inputCompartmentName(cls):
    return 'in'

  @classmethod
  def outputCompartmentName(cls):
    return 'out'

  @classmethod
  def timeOfLastSpikeCompartmentName(cls):
    return 'tols'

  ## Bind Properties to Compartments for ease of use
  @property
  def inputCompartment(self):
    return self.compartments.get(self.inputCompartmentName(), None)

  @inputCompartment.setter
  def inputCompartment(self, inp):
    if inp is not None:
      if inp.shape[1] != self.n_units:
        raise RuntimeError("Input Compartment size does not match provided input size " + str(inp.shape) + "for "
                           + str(self.name))
    self.compartments[self.inputCompartmentName()] = inp

  @property
  def outputCompartment(self):
    return self.compartments.get(self.outputCompartmentName(), None)

  @outputCompartment.setter
  def outputCompartment(self, out):
    if out is not None:
      if out.shape[1] != self.n_units:
        raise RuntimeError("Output compartment size (n, " + str(self.n_units) + ") does not match provided output size "
                           + str(out.shape) + " for " + str(self.name))
    self.compartments[self.outputCompartmentName()] = out

  @property
  def timeOfLastSpike(self):
    return self.compartments.get(self.timeOfLastSpikeCompartmentName(), None)

  @timeOfLastSpike.setter
  def timeOfLastSpike(self, t):
    if t is not None:
      if t.shape[1] != self.n_units:
        raise RuntimeError("Time of last spike compartment size (n, " + str(self.n_units) +
                           ") does not match provided size " + str(t.shape) + " for " + str(self.name))
    self.compartments[self.timeOfLastSpikeCompartmentName()] = t

  # Define Functions
  def __init__(self, name, n_units, max_freq=63.75, key=None, useVerboseDict=False, **kwargs):
    super().__init__(name, useVerboseDict, **kwargs)

    ##Random Number Set up
    self.key = key
    if self.key is None:
      self.key = random.PRNGKey(time.time_ns())

    ## Poisson parameters
    self.max_freq = max_freq ## maximum frequency (in Hertz/Hz)

    ##Layer Size Setup
    self.n_units = n_units
    self.reset()

  def verify_connections(self):
    pass

  def advance_state(self, t, dt, **kwargs):
    self.key, *subkeys = random.split(self.key, 2)
    self.outputCompartment = sample_poisson(subkeys[0], data=self.inputCompartment, dt=dt, fmax=self.max_freq)
    #self.timeOfLastSpike = (1 - self.outputCompartment) * self.timeOfLastSpike + (self.outputCompartment * t)
    self.timeOfLastSpike = update_times(t, self.outputCompartment, self.timeOfLastSpike)

  def reset(self, **kwargs):
    self.inputCompartment = None
    self.outputCompartment = jnp.zeros((1, self.n_units)) #None
    self.timeOfLastSpike = jnp.zeros((1, self.n_units))

  def save(self, **kwargs):
      pass
