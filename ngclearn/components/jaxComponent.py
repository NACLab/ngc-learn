import time

from typing import Union
import jax
from jax import numpy as jnp
from jax import random
from ngcsimlib.compartment import Compartment
from ngcsimlib import Component


class JaxComponent(Component):
    """
    Base Jax component that all Jax-based cells and synapses inherit from.

    Args:
        name: the string name of this cell

        key: PRNG key to control determinism of any underlying random values
            associated with this cell

    """

    def __init__(self, name: str, key: Union[jax.Array, None] = None):
        super().__init__(name)
        self.key = Compartment(
            random.PRNGKey(time.time_ns()) if key is None else key)


    def save(self, directory: str):
        """
        The default save method for JaxComponents, it stores the values of all
        non-targeted (non-wired) compartments into a .npz file.

        Args:
            directory: The directory to save the .npz file.
        """
        file_name = directory + "/" + self.name + ".npz"
        data = {}
        for comp_name, comp in self.compartments:
            if not comp.targeted:
                data[comp_name] = comp.get()
        jnp.savez(file_name, **data)


    def load(self, directory: str):
        """
        The default load method for JaxComponents, it is expected to work with
        the default save. If the save method is modified this one will need to
        be modified too.

        Args:
            directory: The directory to load the .npz file.
        """
        file_name = directory + "/" + self.name + ".npz"
        data = jnp.load(file_name)
        for comp_name, comp in self.compartments:
            d = data.get(comp_name, None)
            if d is not None:
                comp.set(d)

