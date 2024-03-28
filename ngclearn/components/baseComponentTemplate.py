from ngcsimlib.component import Component
from jax import random
import time


class COMPONENT_TEMPLATE(Component):
  ## Class Methods for Compartment Names
  @classmethod
  def DEFAULTCompartmentName(cls):
    return 'DEFAULT'

  ## Bind Properties to Compartments for ease of use
  @property
  def DEFAULTCompartment(self):
    return self.compartments.get(self.DEFAULTCompartmentName(), None)

  @DEFAULTCompartment.setter
  def DEFAULTCompartment(self, x):
    if x is not None:
      if True:
        raise RuntimeError("")
    self.compartments[self.DEFAULTCompartmentName()] = x


  # Define Functions
  def __init__(self, name, key=None, useVerboseDict=False, **kwargs):
    super().__init__(name, useVerboseDict, **kwargs)

    ##Random Number Set up
    self.key = key
    if self.key is None:
      self.key = random.PRNGKey(time.time_ns())

    ##Reset to initialize stuff
    self.reset()

  def verify_connections(self):
    pass

  def advance_state(self, **kwargs):
    pass

  def reset(self, **kwargs):
    pass

  def save(self, directory, **kwargs):
      pass
