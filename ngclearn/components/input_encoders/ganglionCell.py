from ngclearn.components.jaxComponent import JaxComponent
from jax import numpy as jnp, random
from ngclearn import compilable
from ngclearn import Compartment
import jax
from typing import Union, Tuple


def create_gaussian_filter(patch_shape, sigma):
    """
    Create a 2D Gaussian kernel centered on patch_shape with given sigma.
    """
    px, py = patch_shape

    x_ = jnp.linspace(0, px - 1, px)
    y_ = jnp.linspace(0, py - 1, py)

    x, y = jnp.meshgrid(x_, y_)

    xc = px // 2
    yc = py // 2

    filter = jnp.exp(-((x - xc) ** 2 + (y - yc) ** 2) / (2 * (sigma ** 2)))
    return filter


def create_patches(obs, patch_shape, step_shape):
    """
    Extract 2D patches from a batch of images using a sliding window.

    Inputs:
            obs: Input array (B, ix, iy)
            patch_shape: Patch size (px, py)
            step_shape: Stride (sx, sy) -- use 0 for full-overlap

    Output:
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

class RetinalGanglionCell(JaxComponent):
    """
    A groupd of retinal ganglion cell that senses the input
    stimuli and sends out the filtered signal to the brain.

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

        sigma: standard deviation of gaussian kernel

        area_shape: receptive field area of ganglion cells in this module all together

        n_cells: number of ganglion cells in this module

        patch_shape: each ganglion cell receptive field area

        step_shape: the non-overlapping area between each two ganglion cells

        batch_size: batch size dimension of this cell (Default: 1)
    """

    def __init__(self, name: str,
                 filter_type: str,
                 area_shape: Tuple[int, int],
                 n_cells: int,
                 patch_shape: Tuple[int, int],
                 step_shape: Tuple[int, int],
                 batch_size: int = 1,
                 sigma: float = 1.0,
                 key: Union[jax.Array, None] = None,
                 **kwargs):
        super().__init__(name=name, key=key)


        ## Layer Size Setup
        self.filter_type = filter_type
        self.n_cells = n_cells
        self.sigma = sigma

        self.batch_size = batch_size
        self.area_shape = area_shape
        self.patch_shape = patch_shape
        self.step_shape = step_shape

        filter = jnp.ones(self.patch_shape)

        if filter_type == 'gaussian':
            filter = create_gaussian_filter(patch_shape=self.patch_shape, sigma=self.sigma)
        elif filter_type == 'difference_of_gaussian':
            #     TODO: need to be accuarte
            scale = 1.6
            gauss_center = create_gaussian_filter(patch_shape=self.patch_shape, sigma=self.sigma)
            gauss_surround = create_gaussian_filter(patch_shape=self.patch_shape, sigma=self.sigma * scale) * 0.
            filter = gauss_center - gauss_surround #/ (scale**2)


        # ═════════════════ compartments initial values ════════════════════
        in_restVals = jnp.zeros((self.batch_size,
                                 *self.area_shape))    ## input: (B | ix | iy)

        out_restVals = jnp.zeros((self.batch_size,     ## output.shape: (B | n_cells * px * py)
                                  self.n_cells * self.patch_shape[0] * self.patch_shape[1]))

        # ═══════════════════ set compartments ══════════════════════
        self.inputs = Compartment(in_restVals, display_name="Input Stimulus") # input compartment
        self.filter = Compartment(filter, display_name="Filter") # Filter compartment
        self.outputs = Compartment(out_restVals, display_name="Output Signal") # output compartment



    @compilable
    def advance_state(self, t):
        inputs = self.inputs.get()
        filter = self.filter.get()
        px, py = self.patch_shape

        input_patches = create_patches(inputs, patch_shape=self.patch_shape,
                                               step_shape=self.step_shape)

        # ═══════════════════ apply filter to all pathches ══════════════════
        filtered_input = input_patches * filter                        ## shape: (B | n_cell | px | py)

        # ═══════════════════ normalize filtered signals ══════════════════
        filters_output = filtered_input - jnp.mean(filtered_input)      ## shape: (B | n_cell | px | py)

        # ═══════════════════ normalize filtered signals ══════════════════
        outputs = filters_output.reshape(-1, self.n_cells * (px * py))  ## shape: (B | n_cells * px * py)

        self.outputs.set(outputs)

    @compilable
    def reset(self):
        in_restVals = jnp.zeros((self.batch_size,
                                 *self.area_shape))      ## input: (B | ix | iy)

        out_restVals = jnp.zeros((self.batch_size,      ## output.shape: (B | n_cells * px * py)
                                  self.n_cells * self.patch_shape[0] * self.patch_shape[1]))
        #
        # # BUG: the self.inputs here does not have the targeted field
        # # NOTE: Quick workaround is to check if targeted is in the input or not
        # hasattr(self.inputs, "targeted") and not self.inputs.targeted and self.inputs.set(in_restVals)
        self.inputs.set(in_restVals)
        self.outputs.set(out_restVals)

    @classmethod
    def help(cls): ## component help function
        properties = {
            "cell_type": "RetinalGanglionCell - filters the input stimuli, "
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
            "sigma": "Standard deviation of gaussian kernel",
            "area_shape": "Effective receptive field area shape of ganglion cells in this module",
            "n_cells": "Number of Retinal Ganglion (center-surround) cells to model in this layer",
            "patch_shape": "Classical Receptive field area shape of individual ganglion cells in this module",
            "step_shape": "Extra-Classical Receptive field area shape each ganglion cell in this module",
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
        X = RetinalGanglionCell("RGC", filter_type="gaussian",
                                sigma=2.3,
                                area_shape=(16, 26),
                                n_cells = 3,
                                patch_shape=(16, 16),
                                step_shape=(0, 5)
                                )
    print(X)
