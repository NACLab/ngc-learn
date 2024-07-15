from ngclearn import Component, Compartment, numpy as jnp, resolver
from ngcsimlib.logger import warn


class DisplayTile(Component):
    def __init__(self, name, tile_size, highlight_img, goal_img,
                 wall_brightness=100, wall_thickness=1, **kwargs):
        super().__init__(name, **kwargs)

        self.tile_size = tile_size
        self.highlight_img = highlight_img
        self.wall_thickness = wall_thickness
        self.goal_img = goal_img if goal_img is not None else jnp.zeros((tile_size, tile_size), dtype=jnp.uint8)
        self.wall_brightness = wall_brightness

        self.display = Compartment(jnp.zeros((tile_size, tile_size), dtype=jnp.uint8))


        blank = jnp.zeros((self.tile_size, self.tile_size), dtype=jnp.uint8)
        north_wall = blank.at[0:self.wall_thickness, :].set(
            self.wall_brightness)
        south_wall = blank.at[-self.wall_thickness:, :].set(
            self.wall_brightness)
        east_wall = blank.at[:, -self.wall_thickness:].set(self.wall_brightness)
        west_wall = blank.at[:, 0:self.wall_thickness].set(self.wall_brightness)

        # self.displays = [blank, north_wall, south_wall, east_wall, west_wall,
        #                  self.highlight_img, self.goal_img]

        self.displays = jnp.array([north_wall.reshape(tile_size**2),
                                   east_wall.reshape(tile_size ** 2),
                                   south_wall.reshape(tile_size**2),
                                   west_wall.reshape(tile_size**2),
                                   self.highlight_img.reshape(tile_size**2),
                                   self.goal_img.reshape(tile_size**2)])

    def update_display(self, layers):
        _layers = jnp.matmul(jnp.diag(layers), self.displays)
        self.display.set(jnp.max(_layers, axis=0).reshape(self.tile_size, self.tile_size))
