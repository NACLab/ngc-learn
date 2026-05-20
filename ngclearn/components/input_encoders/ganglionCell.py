from ngclearn.components.jaxComponent import JaxComponent
from jax import numpy as jnp, random
from ngclearn import compilable
from ngclearn import Compartment
import jax
from typing import Union, Tuple

def _create_gaussian_filter(patch_shape, sigma):
    ## Create a 2D Gaussian kernel centered on patch_shape with given sigma.
    px, py = patch_shape
    x_ = jnp.linspace(0, px - 1, px)
    y_ = jnp.linspace(0, py - 1, py)
    x, y = jnp.meshgrid(x_, y_)
    xc = px // 2
    yc = py // 2
    _filter = jnp.exp(-((x - xc) ** 2 + (y - yc) ** 2) / (2 * (sigma ** 2)))
    return _filter / jnp.sum(_filter)

def _create_dog_filter(patch_shape, sigma, k=1.6, lmbda=1):
    g1 = _create_gaussian_filter(patch_shape, sigma=sigma)
    g2 = _create_gaussian_filter(patch_shape, sigma=sigma * k)
    dog = g1 - lmbda * g2
    return dog #- jnp.mean(dog)

def _create_patches(obs, patch_shape, step_shape):
    """
    Extract 2D patches from a batch of images using a sliding window.

    Args:
        obs: Input array (B, ix, iy)

        patch_shape: Patch size (px, py)

        step_shape: Stride (sx, sy) -- use 0 for full-overlap

    Returns:
        Patches array (B, n_cells, px, py)

    """

    B, ix, iy = obs.shape
    px, py = patch_shape
    sx, sy = step_shape

    if sx == 0:
        n_x = ix // px
    else:
        n_x = (ix - px) // sx + 1

    if sy == 0:
        n_y = iy // py
    else:
        n_y = (iy - py) // sy + 1

    patches = jnp.stack([
        obs[:,
            i * sx:i * sx + px, j * sy:j * sy + py
            ] for i in range(n_x)
              for j in range(n_y)
    ], axis=1)

    return patches

def _reconstruct(patches, nx_ny, area_shape, patch_shape, step_shape):
    # patches: (N, nx * ny, px, py)

    B = len(patches)
    nx, ny = nx_ny
    ix, iy = area_shape
    px, py = patch_shape
    sx, sy = step_shape
    x = jnp.zeros((B, ix, iy))
    counts = jnp.zeros((ix, iy))

    idx = 0
    for i in range(ny):
        for j in range(nx):
            di = i * sx
            dj = j * sy
            x = x.at[:, di:di + px, dj:dj + py].add(patches[:, idx])
            counts = counts.at[di:di + px, dj:dj + py].add(1.0)
            idx += 1

    return x / counts[None, :, :]



class RetinalGanglionCell(JaxComponent):
    """
    A group of retinal ganglion cell that sense input stimuli and send out filtered
    signals (as output). Note that these simulated cells employ internal generalized 
    filters based on either Gaussian or difference-of-Gaussian kernels) to recover 
    historical receptive field processing effects.

    | --- Cell Input Compartments: ---
    | inputs - input (takes in external signals)
    | --- Cell State Compartments: ---
    | filter - filter (function applied to input)
    | --- Cell Output Compartments: ---
    | outputs - output

    Args:
        name: the string name of this cell

        filter_type: string name of filter function (Default: identity)
            :Note: supported filters include "gaussian", "difference_of_gaussian"

        sigma: standard deviation of (gaussian) kernel

        area_shape: shape of receptive field area of ganglion cells in this module (all together)

        n_cells: number of ganglion cells in this module

        patch_shape: shape of each ganglion cell's receptive field area

        step_shape: the non-overlapping area between each pair (two) of ganglion cells

        batch_size: batch size dimension of this cell/module (Default: 1)
    """

    def __init__(
        self, 
        name: str,
        filter_type: str,
        area_shape: Tuple[int, int],
        n_cells: int,
        patch_shape: Tuple[int, int],
        step_shape: Tuple[int, int],
        batch_size: int = 1,
        sigma: float = 1.0,
        key: Union[jax.Array, None] = None,
        **kwargs
    ):
        super().__init__(name=name, key=key)

        ## Layer Size Setup
        self.filter_type = filter_type
        self.n_cells = n_cells
        self.sigma = sigma

        self.batch_size = batch_size
        self.area_shape = area_shape
        self.patch_shape = patch_shape
        self.step_shape = step_shape

        _filter = jnp.ones(self.patch_shape)
        if filter_type == 'gaussian':
            _filter = _create_gaussian_filter(patch_shape=self.patch_shape, sigma=self.sigma)
        elif filter_type == 'difference_of_gaussian':
            _filter = _create_dog_filter(patch_shape=self.patch_shape, sigma=sigma)

        # ═════════════════ compartments initial values ════════════════════
        in_restVals = jnp.zeros((batch_size, *self.area_shape)) ## input: (B | ix | iy)

        out_restVals = jnp.zeros(
            (batch_size, self.n_cells * self.patch_shape[0] * self.patch_shape[1])
        ) ## output.shape: (B | n_cells * px * py)

        # ═══════════════════ set compartments ══════════════════════
        self.inputs = Compartment(in_restVals, display_name="Input Stimulus") # input compartment
        self.filter = Compartment(_filter, display_name="Filter") # Filter compartment
        self.outputs = Compartment(out_restVals, display_name="Output Signal") # output compartment

    @compilable
    def advance_state(self, t):
        inputs = self.inputs.get()
        _filter = self.filter.get()
        px, py = self.patch_shape

        # ═══════════════════ extract pathches for filters ══════════════════
        input_patches = _create_patches(inputs, patch_shape=self.patch_shape, step_shape=self.step_shape)

        # ═══════════════════ apply filter to all pathches ══════════════════
        filtered_input = input_patches * _filter                                 ## shape: (B | n_cells | px | py)

        # ════════════ reshape all cells responses to a single input to brain ════════════
        filtered_input = filtered_input.reshape(-1, self.n_cells * (px * py))   ## shape: (B | n_cells * px * py)

        # ═══════════════════ normalize filtered signals ══════════════════
        outputs = filtered_input - jnp.mean(filtered_input, axis=1, keepdims=True)         ## shape: (B | n_cells * px * py)

        self.outputs.set(outputs)



    @compilable
    def reset(self):  ## reset core components/statistics
        in_restVals = jnp.zeros((self.batch_size, *self.area_shape))      ## input: (B | ix | iy)
        out_restVals = jnp.zeros((self.batch_size,      ## output.shape: (B | n_cells * px * py)
                                  self.n_cells * self.patch_shape[0] * self.patch_shape[1]))
        self.inputs.set(in_restVals)
        self.outputs.set(out_restVals)

    @classmethod
    def help(cls): ## component help function
        properties = {
            "cell_type": "RetinalGanglionCell - filters the input stimuli according retinal ganglion dynamics"
        }
        compartment_props = {
            "inputs":
                {"inputs": "Takes in external input signal values"},
            "states":
                {"filter": "Preprocessing function applies to input)"},
            "outputs":
                {"outputs": "Preprocessed signal values emitted at time t"},
        }
        hyperparams = {
            "filter_type": "Type of the filter for preprocessing the input",
            "sigma": "Standard deviation of gaussian kernel/filter",
            "area_shape": "Effective receptive field area shape of ganglion cells in this module",
            "n_cells": "Number of retinal ganglion (center-surround) cells to model in this layer",
            "patch_shape": "Classical receptive field area shape of individual ganglion cells in this module",
            "step_shape": "Extra-classical receptive field area shape each ganglion cell in this module",
            "batch_size": "Batch size dimension of this component"
        }
        info = {cls.__name__: properties,
                "compartments": compartment_props,
                "dynamics": "~ Gaussian(x)",
                "hyperparameters": hyperparams}
        return info

if __name__ == '__main__':
    from ngcsimlib.context import Context
    with Context("Bar") as bar:
        X = RetinalGanglionCell(
                "RGC", 
                filter_type="gaussian",
                sigma=2.3,
                area_shape=(16, 26),
                n_cells = 3,
                patch_shape=(16, 16),
                step_shape=(0, 5)
        )
    print(X)




