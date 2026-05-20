import jax.numpy as jnp
from jax import lax, jit
from functools import partial

def _calc_gaussian_kernel_2D( ## internal co-routine
        sigma: float,
        radius: int
) -> jnp.ndarray:
    ## Generate a (normalized) 2D Gaussian kernel of shape: (1, 1, 2*radius+1, 2*radius+1)
    x = jnp.arange(-radius, radius + 1)
    xx, yy = jnp.meshgrid(x, x)
    kernel = jnp.exp(-0.5 * (xx ** 2 + yy ** 2) / sigma ** 2)
    kernel = kernel / jnp.sum(kernel)
    # Reshape to (out_channels, in_channels, height, width) for lax.conv_general_dilated
    return kernel[jnp.newaxis, jnp.newaxis, :, :]

@partial(jit, static_argnums=[3, 4])
def gaussian_filter(
        images: jnp.ndarray, ## input image batch
        sigma_center: float, ## sigma1
        sigma_surround: float, ## sigma2
        kernel_size : int, ## radius
        use_ratio=False, ## if True, this becomes a ratio-of-Gaussians
        edge_pad_mode="edge" ## "reflect"
) -> jnp.ndarray:
    """
    Applies a difference-of-Gaussians filter to a batch of 2D images (of CxHxW tensor shape).

    Args:
        images: Input image tensor of shape (B, C, H, W)

        sigma_center: Standard deviation for narrow / center blur

        sigma_surround: Standard deviation for wide / surround blur

        kernel_size: Kernel radius (window size will be `2*radius + 1`)

        use_ratio: whether or not to use a ratio-of-Gaussians filter (Default: False)

    Returns:
        An output tensor of shape (B, C, H, W)
    """
    ## Pad spatial dimensions (H, W) using edge-clamping to remove artifacts
    # Format for 4D (B, C, H, W): ((Before_B, After_B), (Before_C, After_C), (Before_H, After_H), (Before_W, After_W))
    padding_config = ((0, 0), (0, 0), (kernel_size, kernel_size), (kernel_size, kernel_size))
    padded_x = jnp.pad(images, padding_config, mode=edge_pad_mode)

    ## Construct two 2D Gaussian kernels
    k1 = _calc_gaussian_kernel_2D(sigma_center, kernel_size) ## center kernel
    k2 = _calc_gaussian_kernel_2D(sigma_surround, kernel_size) ## surround kernel

    ## Define dimension ordering for lax.conv ('NCHW' standard layout)
    dn = lax.ConvDimensionNumbers(
        lhs_spec=(0, 1, 2, 3), ## (batch, channel, height, width)
        rhs_spec=(0, 1, 2, 3), ## (out_channel, in_channel, height, width)
        out_spec=(0, 1, 2, 3)  ## (batch, channel, height, width)
    )

    ## Extract channel count dynamically for independent channel-wise filtering
    num_channels = images.shape[1]

    ## Perform spatial convolutions w/ 'VALID' padding on the edge-padded input
    blur_center = lax.conv_general_dilated(
        padded_x, k1, window_strides=(1, 1), padding='VALID', dimension_numbers=dn, feature_group_count=num_channels
    )
    blur_surround = lax.conv_general_dilated(
        padded_x, k2, window_strides=(1, 1), padding='VALID', dimension_numbers=dn, feature_group_count=num_channels
    )

    ## Perform final filter calculation
    if use_ratio:
        eps = 1e-5
        output = blur_center / (blur_surround + eps) ## Compute kernel ratio
    else:
        output = blur_center - blur_surround ## Compute kernel difference
    return output ## shape: (B, C, H, W)

