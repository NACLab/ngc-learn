"""
Patch sampler system.
"""
import random
import math
from jax import numpy as jnp

class RandomPatchGenerator:
    """
    Patch sampler (for 2D grids, e.g., images), in Jax!

    @author Will Gebhardt
    """
    def __init__(self, image_shape, patch_size, slices=None):
        rows, cols = image_shape
        self.regions = []
        if slices is None:
            self.regions.append(((0, 0), (rows - patch_size, cols - patch_size)))
        else:
            row_step = (rows - patch_size + 1) / (slices + 1)
            cols_step = (cols - patch_size + 1) / (slices + 1)
            for r in range(slices + 1):
                for c in range(slices + 1):
                  upper_left_bound = (math.floor(row_step * r),
                                      math.floor(cols_step * c))
                  lower_right_bound = (math.floor(row_step * (r + 1)) - 1,
                                       math.floor(cols_step * (c + 1)) - 1)
                  self.regions.append((upper_left_bound, lower_right_bound))
            self.loc = 0
            self.patch_size = patch_size


    def __next__(self):
        (r_min, c_min), (r_max, c_max) = self.regions[self.loc]
        r = r_min if r_min == r_max else random.randint(r_min, r_max)
        c = c_min if c_min == c_max else random.randint(c_min, c_max)
        # s = [(0, 0), (0, 14), (14, 0), (14, 14)]
        # r, c = s[self.loc]
        self.loc += 1
        if self.loc >= len(self.regions):
            self.reset()
        return r, c

    def get_patches(self, image, count, reset=False, pStimulus=0.2):
        if reset:
            self.reset()
        patch_set = jnp.zeros([count, self.patch_size * self.patch_size])
        #patch_set = tf.Variable(tf.zeros((count, self.patch_size * self.patch_size)))
        for i in range(count):
            r, c = next(self)
            p = jnp.reshape(image[r:r+self.patch_size, c:c+self.patch_size],
                            [self.patch_size * self.patch_size])
            # p = tf.reshape(tf.Variable(image[r:r+self.patch_size, c:c+self.patch_size], dtype=tf.float32),
            #                            [1, self.patch_size * self.patch_size])
        #while tf.reduce_sum(p) < 0.2 * (self.patch_size * self.patch_size):
        n_attempts = 0
        while jnp.sum(p) < pStimulus * (self.patch_size * self.patch_size):
            r, c = next(self)
            p = jnp.reshape(image[r:r + self.patch_size, c:c + self.patch_size],
                            [self.patch_size * self.patch_size])
            # p = tf.reshape(tf.Variable(image[r:r + self.patch_size, c:c + self.patch_size], dtype=tf.float32),
            #                [1, self.patch_size * self.patch_size])
            n_attempts += 1
            if n_attempts >= self.get_region_count():
                break
        patch_set = patch_set.at[i, :].set(p) #patch_set[i, :].assign(p)
        return patch_set

    def get_coverage(self, image, patches_per_region, pStimulus=0.2):
        return self.get_patches(image, patches_per_region * len(self.regions),
                                reset=True, pStimulus=pStimulus)

    def reset(self):
        self.loc = 0
        random.shuffle(self.regions)

    def get_region_count(self):
        return len(self.regions)


# patch_generator = RandomPatchGenerator(model_hyper_params['image_shape'], patch_size, slices)
# sample = patch_generator.get_coverage(square_image, batches)
# trials = tf.random.uniform(sample.shape, 0., 1.)
