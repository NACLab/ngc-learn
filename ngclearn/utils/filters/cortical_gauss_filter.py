"""
Support for a function/pure "cortical" Gaussian filter - this can be configured to facilitate a filter that
engages in a difference-of-Gaussians-like or ratio-of-Gaussians-like process
"""

import jax.numpy as jnp
from jax import lax, jit
from functools import partial

def _calc_gaussian_kernel_2D(
        sigma: float, ## standard deviation of kernel
        radius: int ## controls shape of kernel
) -> jnp.ndarray: ## internal co-routine for Gaussian kernel
    ## create a normalized 2D Gaussian kernel with shape (1, 1, 2*radius+1, 2*radius+1)
    x = jnp.arange(-radius, radius + 1)
    xx, yy = jnp.meshgrid(x, x)
    kernel = jnp.exp(-0.5 * (xx ** 2 + yy ** 2) / sigma ** 2)
    kernel = kernel / jnp.sum(kernel)
    return kernel[jnp.newaxis, jnp.newaxis, :, :]

@partial(jit, static_argnums=[3, 4, 5])
def cortical_gaussian_filter(
    images: jnp.ndarray, ## expected input shape : (B, C, H, W)
    sigma_center: float, ## center excitation
    sigma_surround: float, ## surround inhibition
    kernel_size: int, ## kernel radius
    use_ratio:bool=False, ## triggers either RoG vs DoG mode
    semi_sat_constant:float=0.1, ## sigma_h (controls contrast-gain)
    excitation_exp:float=2.0, ## p-exponent (typically in range of 1.5 - 2.0)
    inhibition_exp:float=2.0, ## q-exponent
    edge_pad_mode:str='edge'
) -> jnp.ndarray:
    """
    Applies a configurable rectified Gaussian filter (either difference-of-Gaussians, i.e., DoG, or ratio-of-Gaussians,
    i.e., RoG) to a tensor batch of 2D images (each of CxHxW tensor shape/format). Note that this variant filter
    means that DoG mode acts more as a (half-wave) rectified subtraction of two Gaussian kernels and RoG mode acts more
    as simple form of divisive normalization.

    Args:
        images: input image tensor of shape (B, C, H, W)

        sigma_center: standard deviation for narrow / center blur

        sigma_surround: standard deviation for wide / surround blur

        kernel_size: kernel radius (window size will be `2*radius + 1`)

        use_ratio: if True, this filter applies a ratio-of-Gaussians (RoG) filter (Default: False)

        semi_sat_constant: suppresses amplified micro-variations in sensory/image space

        excitation_exp: p-exponent (typically in range of 1.5 - 2.0) (Default: 2.0)

        inhibition_exp: q-exponent (typically in range of 1.5 - 2.0) (Default: 2.0)

        edge_pad_mode: type of image edge-clamping/padding to use, either "edge" or "reflect" (Default: "edge")

    Returns:
        An output tensor of shape (B, C, H, W)
    """
    ## set up edge-artifact padding correction
    padding_config = ((0, 0), (0, 0), (kernel_size, kernel_size), (kernel_size, kernel_size))
    padded_x = jnp.pad(images, padding_config, mode=edge_pad_mode)

    ## set up Gaussian kernels
    k1 = _calc_gaussian_kernel_2D(sigma_center, kernel_size)
    k2 = _calc_gaussian_kernel_2D(sigma_surround, kernel_size)

    dn = lax.ConvDimensionNumbers(
        lhs_spec=(0, 1, 2, 3), rhs_spec=(0, 1, 2, 3), out_spec=(0, 1, 2, 3)
    )
    num_channels = images.shape[1]

    ## apply standard spatial convolutions
    blur_center = lax.conv_general_dilated(
        padded_x, k1, window_strides=(1, 1), padding='VALID', dimension_numbers=dn, feature_group_count=num_channels
    )
    blur_surround = lax.conv_general_dilated(
        padded_x, k2, window_strides=(1, 1), padding='VALID', dimension_numbers=dn, feature_group_count=num_channels
    )
    if use_ratio:
        ## cortically-plausible divisive normalization (or a cortical ratio-of-Gaussians; RoG)
        ### first, apply half-wave rectification
        rectified_center = jnp.maximum(0.0, blur_center)
        rectified_surround = jnp.maximum(0.0, blur_surround)
        ## next, apply nonlinear response exponents
        numerator = jnp.power(rectified_center, excitation_exp)
        denominator = jnp.power(rectified_surround, inhibition_exp) + (semi_sat_constant ** 2)
        output = numerator / denominator ## calculate ratio
    else: ## cortically-plausible difference-of-Gaussians (cortical DoG)
        ### this is modeled via rectified linear subtraction under an output threshold (0 in the case below)
        output = jnp.maximum(0.0, blur_center - blur_surround)
    return output

