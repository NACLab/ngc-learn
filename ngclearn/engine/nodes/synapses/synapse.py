from ngclearn.engine.nodes.node import Node
from abc import ABC

class Synapse(Node, ABC):
    """
    Base synapse element (class from which other synapse types inherit basic
    properties from). This is an explicit node representation of a transform
    applied across a cable (since cables between nodes are implicit graph
    constructs).

    Args:
        name: the string name of this cable

        shape: the tensor shape (minimum is 2D) of this synaptic cable

        dt: integration time constant

        key: PRNG Key to control determinism of any underlying synapses
            associated with this cable
    """
    def __init__(self, name, shape, dt, key=None):
        super().__init__(name=name, dt=dt, key=key)
        self.shape = shape

    def evolve(self):
        """
        Execute this synapses's internal plasticity calculation scheme and
        physically adapt the value of any internal parameters inherent to this
        synaptic cable, i.e., this function call runs this synapse's plasticity
        for one step in time.
        """
        pass

class_name = Synapse.__name__
