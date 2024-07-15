from ngclearn import Compartment, Component, numpy as jnp
from ngcsimlib.utils import add_component_resolver, add_resolver_meta, get_current_path


class Screen(Component):
    auto_resolve = False
    def __init__(self, name, width, height, tile_size, **kwargs):
        super().__init__(name, **kwargs)
        self.width = width
        self.height = height
        self.tile_size = tile_size
        self._compartments = []

        self.display = Compartment(
            jnp.zeros((tile_size * width, tile_size * height), dtype=jnp.uint8))

        self.inputs = []
        for y in range(height):
            self.inputs.append([])
            for x in range(width):
                _c = Compartment(jnp.zeros((tile_size, tile_size), dtype=jnp.uint8))
                self.__dict__[f"{name}_{y}_{x}"] = _c
                self.inputs[y].append(_c)
                self._compartments.append((f"{name}_{y}_{x}", y, x))

    @staticmethod
    def build_advance_state(component):
        compartments = component._compartments
        tile_size = component.tile_size
        @staticmethod
        def _advance(display, **kwargs):
            for c, y, x in compartments:
                display = display.at[y * tile_size:(y + 1) * tile_size,
                          x * tile_size:(x + 1) * tile_size].set(kwargs.get(c))
            return display

        return _advance, ["display"], [], [], ["display"] + [c for c, _, _ in compartments]