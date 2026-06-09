"""
Support for a function/pure Gaussian filter - this can be configured to facilicate
difference-of-Gaussians (DoG) or ratio-of-Gaussians (RoG).
"""

import jax.numpy as jnp
from jax import lax, jit
from functools import partial

def _calc_gaussian_kernel_2D( ## internal co-routine for Gaussian kernel
        sigma: float, ## standard deviation of kernel
        radius: int ## controls shape of kernel
) -> jnp.ndarray:
    ## create a normalized 2D Gaussian kernel with shape (1, 1, 2*radius+1, 2*radius+1)
    x = jnp.arange(-radius, radius + 1)
    xx, yy = jnp.meshgrid(x, x)
    kernel = jnp.exp(-0.5 * (xx ** 2 + yy ** 2) / sigma ** 2)
    kernel = kernel / jnp.sum(kernel)
    ## reshape output to: (out_channels, in_channels, height, width) to support lax.conv_general_dilated w/in filter
    return kernel[jnp.newaxis, jnp.newaxis, :, :]

@partial(jit, static_argnums=[3, 4])
def gaussian_filter( ## core filter routine
        images: jnp.ndarray, ## input image batch
        sigma_center: float, ## sigma1
        sigma_surround: float, ## sigma2
        kernel_size : int, ## radius
        use_ratio:bool=False, ## if True, this becomes a ratio-of-Gaussians
        edge_pad_mode:str="edge" ## "reflect"
) -> jnp.ndarray:
    """
    Applies a configurable Gaussian filter (either difference-of-Gaussians or ratio-of-Gaussians) to a tensor batch of
    2D images (of CxHxW tensor shape).

    Args:
        images: input image tensor of shape (B, C, H, W)

        sigma_center: standard deviation for narrow / center blur

        sigma_surround: standard deviation for wide / surround blur

        kernel_size: kernel radius (window size will be `2*radius + 1`)

        use_ratio: if True, this filter applies a ratio-of-Gaussians (RoG) filter (Default: False)

        edge_pad_mode: type of image edge-clamping/padding to use, either "edge" or "reflect" (Default: "edge")

    Returns:
        An output tensor of shape (B, C, H, W)
    """
    ## pad spatial dimensions (H, W) using edge/reflect-clamping in order to remove artifacts
    ### format for 4D tensor (B, C, H, W) =>
    ###   result: ((Before_B, After_B), (Before_C, After_C), (Before_H, After_H), (Before_W, After_W))
    padding_config = ((0, 0), (0, 0), (kernel_size, kernel_size), (kernel_size, kernel_size))
    padded_x = jnp.pad(images, padding_config, mode=edge_pad_mode)

    ## construct two 2D Gaussian kernels
    k1 = _calc_gaussian_kernel_2D(sigma_center, kernel_size) ## center kernel
    k2 = _calc_gaussian_kernel_2D(sigma_surround, kernel_size) ## surround kernel

    ## define dimension ordering for lax.conv ('NCHW' standard layout)
    dn = lax.ConvDimensionNumbers(
        lhs_spec=(0, 1, 2, 3), ## (batch, channel, height, width)
        rhs_spec=(0, 1, 2, 3), ## (out_channel, in_channel, height, width)
        out_spec=(0, 1, 2, 3)  ## (batch, channel, height, width)
    )
    num_channels = images.shape[1] ## get channel count dynamically for independent channel-wise filtering

    ## below performs spatial convolutions w/ "VALID" padding on the edge-padded input
    blur_center = lax.conv_general_dilated( ## center Gaussian
        padded_x,
        k1,
        window_strides=(1, 1),
        padding='VALID',
        dimension_numbers=dn,
        feature_group_count=num_channels
    )
    blur_surround = lax.conv_general_dilated( ## surround Gaussian
        padded_x,
        k2,
        window_strides=(1, 1),
        padding='VALID',
        dimension_numbers=dn,
        feature_group_count=num_channels
    )
    ## Perform final filter calculation
    if use_ratio: ## apply ratio-of-Gaussians (RoG)
        eps = 1e-5
        output = blur_center / (blur_surround + eps) ## calculate kernel ratio
    else: ## apply difference-of-Gaussians (DoG)
        output = blur_center - blur_surround ## calculate kernel difference
    return output ## final shape: (B, C, H, W)

