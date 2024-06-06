import time
from jax import random
from ngclearn import resolver, Component, Compartment

class JaxComponent(Component):
    """
    Base Jax component that all Jax-based cells and synapses inherit from.

    Args:
        name: the string name of this cell

        key: PRNG key to control determinism of any underlying random values
            associated with this cell

        directory: string indicating directory on disk to save component parameter
            values to
    """

    def __init__(self, name, key=None, directory=None, **kwargs):
        super().__init__(name, **kwargs)
        self.directory = directory
        self.key = Compartment(
            random.PRNGKey(time.time_ns()) if key is None else key)

