from ngclearn import Context, numpy as jnp
from ngclearn.evironment.displayTile import DisplayTile
from ngclearn.evironment.screen import Screen
from ngcsimlib.logger import warn

class World(Context):
    def __init__(self, name, world_width, world_height, view_width=None,
                 view_height=None):
        super().__init__(name)

        self.width = world_width
        self.height = world_height
        self.view_width = view_width if view_width is not None else world_width
        self.view_height = view_height if view_height is not None else (
            world_height)
        if view_width is None and view_height is None:
            self.ego = False
        else:
            self.ego = True

        self.movement_map = jnp.zeros((self.height, self.width))

        self._tiles = None
        self.screen = None
        self._wall_map = None

        self._current_position = (0, 0)
        self._goal_position = (0, 0)

    def reset(self, start_loc):
        self._current_position = start_loc
        self._update_position()

    @property
    def current_position(self):
        return self._current_position

    @property
    def goal_position(self):
        return self._goal_position

    def set_movement(self, locs, movable=True):
        for y, x in locs:
            self.movement_map = self.movement_map.at[y, x].set(1 if movable else 0)


    def _build_tiles(self, **kwargs):
        for y in range(self.view_height):
            self._tiles.append([])
            for x in range(self.view_width):
                new_tile = DisplayTile(name=f"t_{y}_{x}", **kwargs)
                self._tiles[y].append(new_tile)
                self.screen.inputs[y][x] << new_tile.display

    def _build_wall_map(self):
        wall_map = jnp.zeros((self.height, self.width, 4), jnp.uint8)
        for y in range(self.height):
            for x in range(self.width):
                movable_tile = self.movement_map[y][x]
                if movable_tile == 0:
                    continue

                #North
                dy = y - 1
                if dy < 0:
                    wall_map = wall_map.at[y, x, 0].set(1)
                elif self.movement_map[dy][x] == 0:
                    wall_map = wall_map.at[y, x, 0].set(1)

                #east
                dx = x + 1
                if dx >= self.width:
                    wall_map = wall_map.at[y, x, 1].set(1)
                elif self.movement_map[y][dx] == 0:
                    wall_map = wall_map.at[y, x, 1].set(1)

                #South
                dy = y + 1
                if dy >= self.height:
                    wall_map = wall_map.at[y, x, 2].set(1)
                elif self.movement_map[dy][x] == 0:
                    wall_map = wall_map.at[y, x, 2].set(1)

                #West
                dx = x - 1
                if dx < 0:
                    wall_map = wall_map.at[y, x, 3].set(1)
                elif self.movement_map[y][dx] == 0:
                    wall_map = wall_map.at[y, x, 3].set(1)
        self._wall_map = wall_map

    def get_view_radi(self): # North radius, East radius, South Radius, West radius
        ev = wv = (self.view_width-1) // 2
        nv = sv = (self.view_height-1) // 2
        return -nv, ev, sv, -wv

    def _update_tile(self, dy, dx, ay, ax):
        if 0 <= ay < self.height and 0 <= ax < self.width:
            layers = jnp.array([0, 0, 0, 0, 0, 0], dtype=jnp.uint8)
            if (ay, ax) == self._current_position:
                layers = layers.at[4].set(1)
            if (ay, ax) == self._goal_position:
                layers = layers.at[5].set(1)

            layers = layers.at[:4].set(self._wall_map[ay, ax, :])
            self._tiles[dy][dx].update_display(layers)
        else:
            layers = jnp.array([0, 0, 0, 0, 0, 0], dtype=jnp.uint8)
            self._tiles[dy][dx].update_display(layers)

    def _update_position(self, old_poc=None):
        if self.ego:

            nr, er, sr, wr = self.get_view_radi()
            cy, cx = self._current_position
            for ty, y in enumerate(range(nr, sr+1)):
                for tx, x in enumerate(range(wr, er+1)):
                    self._update_tile(ty, tx, cy+y, cx+x)

        else:
            if old_poc is None:
                for y in range(self.height):
                    for x in range(self.width):
                        self._update_tile(y, x, y, x)
            else:
                oy, ox = old_poc
                self._update_tile(oy, ox, oy, ox)
                ny, nx = self._current_position
                self._update_tile(ny, nx, ny, nx)

    def initialize(self, tile_size,
                   highlight_img,
                   goal_img,
                   wall_brightness=100,
                   wall_thickness=1,
                   player_location=(0, 0),
                   goal_location=(0, 0)):

        if self._tiles is not None:
            warn(f"{self.name} is already initialized, skipping")
            return

        self._current_position = player_location
        self._goal_position = goal_location

        self._tiles = []
        self.screen = Screen("screen", height=self.view_height,
                              width=self.view_width, tile_size=tile_size)
        self._build_tiles(tile_size=tile_size, highlight_img=highlight_img,
                          goal_img=goal_img, wall_brightness=wall_brightness,
                          wall_thickness=wall_thickness)
        self._build_wall_map()

        self._update_position()



    def move(self, action):
        dy, dx = action
        ly, lx = self._current_position

        ny = ly + dy
        nx = lx + dx

        if 0 <= ny < self.height and 0 <= nx < self.width:
            if self.movement_map[ny][nx] != 0:
                self._current_position = (ny, nx)
        else:
           return
        self._update_position((ly, lx))



