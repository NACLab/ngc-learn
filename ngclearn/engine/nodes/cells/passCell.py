## pass-through cell
from ngclearn.engine.nodes.cells.cell import Cell

class PassCell(Cell):  # inherits from Node class
    """
    A simple input (pass-through) cell

    Args:
        name: the string name of this cell

        n_units: number of cellular entities (neural population size)

        dt: integration time constant

        seed: integer seed to control determinism of any underlying synapses
            associated with this cell
    """
    def __init__(self, name, n_units, dt, seed=69):
        super().__init__(name, n_units, dt, seed)
        # cell compartments
        self.comp["in"] = None

    def step(self):
        self.t = self.t + self.dt
        self.gather()
        # x = self.comp.get("in") # get input stimulus
        self.comp['out'] = self.comp.get("in")

    # def make_callback(self, comp_name):
    #     def callback():
    #         print(self.comp)
    #         return self.comp[comp_name]
    #
    #     return callback
    #
    #     # return lambda : self.comp[comp_name]

class_name = PassCell.__name__
