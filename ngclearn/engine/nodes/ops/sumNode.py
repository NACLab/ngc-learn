## pass-through cell
from ngclearn.engine.nodes.ops.op import Op
from ngclearn.engine.utils.bundle_rules import additive

class SumNode(Op):  # inherits from Op class
    """
    A summation node -- sums all signals from relevant compartments according
    to its additive bundle rule.

    Args:
        name: the string name of this operator

        n_units: number of calculating entities or units

        dt: integration time constant

        key: PRNG Key to control determinism of any underlying synapses
            associated with this operator
    """
    def __init__(self, name, n_units, dt, key=None, debugging=False):
        super().__init__(name, n_units, dt, key, debugging=debugging)
        self.add_bundle_rule('input', additive(self))

    def pre_gather(self):
        self.comp['in'] = 0

    def step(self):
        self.t = self.t + self.dt
        self.gather()
        self.comp['out'] = self.comp['in']

class_name = SumNode.__name__
