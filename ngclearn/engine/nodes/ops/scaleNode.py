from ngclearn.engine.nodes.ops.op import Op


class ScaleNode(Op):  # inherits from Node class
    """
    A scaling node

    Args:
        name: the string name of this operator

        n_units: number of calculating entities or units

        dt: integration time constant

        scale: scaling factor to apply to this node's output (compartment)

        seed: integer seed to control determinism of any underlying synapses
            associated with this operator
    """
    def __init__(self, name, n_units, dt, scale, seed=69):
        super().__init__(name, n_units, dt, seed)
        self.scale = scale

    def step(self):
        self.t = self.t + self.dt
        self.gather()
        self.comp['out'] = self.scale * (1 - self.comp['in']) ## hacky hard-coded inversion

    def custom_dump(self, node_directory, template=False) -> dict[str, any]:
        required_keys = {'scale': self.scale}
        return {**super().custom_dump(node_directory, template),
                **{k: self.__dict__.get(k, None) for k in required_keys}}

class_name = ScaleNode.__name__
