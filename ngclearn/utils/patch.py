from typing import Literal

from jax import numpy as jnp

from ngcsimlib.logger import error, warn


class PatchGenerator(object):
    def __init__(self,
                 patch_height: int,
                 patch_width: int,
                 horizontal_alignment: Literal['left', 'right', 'center', 'fit']=None,
                 vertical_alignment: Literal['top', 'bottom', 'center', 'fit']=None,
                 horizontal_stride: int | None = None,
                 vertical_stride: int | None = None):
        self.horizontal_alignment = horizontal_alignment or 'left'
        self.horizontal_stride = horizontal_stride or 0
        self.patch_height = patch_height

        self.vertical_alignment = vertical_alignment or 'top'
        self.vertical_stride = vertical_stride or 0
        self.patch_width = patch_width

        self.idx_cache = {}

        self._current_height = None
        self._current_width = None

        self._max_patch = None
        self._current_idx = -1
        self._current_img = None

    def __iter__(self):
        if self._current_img is None:
            error("Attempting to generate patches but no image has been provided")

        self._current_idx = 0
        return self

    def target(self, img: jnp.ndarray):
        height, width = img.shape[:2]
        if height == self._current_height and width == self._current_width:
            self._current_img = img
            return

        if self.patch_height > height or self.patch_width > width:
            warn("Image to small for patches to be extracted, aborting")
            return

        horizontal_idxs = []
        vertical_idxs = []

        actual_patch_width = self.patch_width - self.horizontal_stride
        if self.horizontal_alignment == 'left':
            horizontal_idxs += range(0, width-self.patch_width, actual_patch_width)
        elif self.horizontal_alignment == 'right':
            horizontal_idxs += [i - self.patch_width for i in range(width, self.patch_width, -actual_patch_width)]
        elif self.horizontal_alignment == 'center':
            centerx = width // 2
            horizontal_idxs += range(centerx, width-self.patch_width, actual_patch_width)
            horizontal_idxs += [i - self.patch_width for i in range(centerx, self.patch_width, -actual_patch_width)]
        elif self.horizontal_alignment == 'fit':
            extra = ((width - self.patch_width) % actual_patch_width) // 2
            horizontal_idxs += range(extra, width - self.patch_width + 1,
                                     actual_patch_width)
        else:
            pass

        actual_patch_height = self.patch_height - self.vertical_stride
        if self.vertical_alignment == 'left':
            horizontal_idxs += range(0, height-self.patch_height, actual_patch_height)
        elif self.vertical_alignment == 'right':
            horizontal_idxs += [i - self.patch_height for i in range(height, self.patch_width, -actual_patch_height)]
        elif self.vertical_alignment == 'center':
            centery = height // 2
            horizontal_idxs += range(centery, height-self.patch_height, actual_patch_height)
            horizontal_idxs += [i - self.patch_width for i in range(centery, self.patch_height, -actual_patch_height)]
        elif self.vertical_alignment == 'fit':
            extra = ((height - self.patch_height) % actual_patch_height) // 2
            horizontal_idxs += range(extra, height - self.patch_height + 1,
                                     actual_patch_height)

        print(horizontal_idxs)

        img = jnp.zeros((len(horizontal_idxs), width))
        for row, idx in enumerate(horizontal_idxs):
            img = img.at[row, idx:idx + self.patch_width].set(
                img[row, idx:idx + self.patch_width] + 50)

        import matplotlib.pyplot as plt

        plt.imshow(img)
        plt.show()


## testing code
# gen = PatchGenerator(patch_width=5, patch_height=5, horizontal_alignment='center', horizontal_stride=1)
#
# test_img = jnp.zeros((32, 32))
#
# gen.target(test_img)
