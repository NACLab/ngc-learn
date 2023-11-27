from ngclearn.engine.nodes.cells.cell import Cell

class ErrCell(Cell):  # inherits from Node class
    """
    A simple (non-spiking) error cell - this is a rate-coded approximation of a
    mismatch signal.

    Args:
        name: the string name of this cell

        n_units: number of cellular entities (neural population size)

        dt: integration time constant

        key: PRNG Key to control determinism of any underlying synapses
            associated with this cell
    """
    def __init__(self, name, n_units, dt, key=None, debugging=False):
        super().__init__(name, n_units, dt, key, debugging=debugging)
        # cell compartments
        self.comp["err"] = None
        self.comp["targ"] = None
        self.comp["pred"] = None

    def step(self):
        self.t = self.t + self.dt
        self.gather()
        mu = self.comp["pred"]
        y = self.comp["targ"]
        self.comp["err"] = (y - mu)  # calc error / mismatch

    @staticmethod
    def get_default_in():
        """
        Returns the value within input compartment ``pred``
        """
        return 'pred'

    @staticmethod
    def get_default_out():
        """
        Returns the value within output compartment ``err``
        """
        return 'err'

class_name = ErrCell.__name__
