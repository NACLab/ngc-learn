"""
Calculation toolbox that drives conv/deconv operations in the ngc-learn
convolution components sub-branch; this contains routines/co-routines
for `ngclearn.components.synapses.convolution`.
"""
import numpy as np
from jax import jit, numpy as jnp, random, nn, lax
from functools import partial
from jax._src import core

@partial(jit, static_argnums=[1])
def _pad(x, padding):
    """
    Jit-i-fied padding function.

    Args:
        x (ndarray): The input array to be padded.

        padding (tuple): A tuple containing the amounts of padding to apply to each dimension;
            Format: (pad_bottom, pad_top, pad_left, pad_right).

    Returns:
        _x (ndarray): The padded array.
    """
    
    # Unpack the padding amounts for each dimension
    pad_bottom, pad_top, pad_left, pad_right = padding

    # Apply padding to the input array 'x'
    # Padding is applied as:
    # [[0, 0],           # No padding to the batch dimension
    #  [pad_bottom, pad_top], # Padding for the height dimension
    #  [pad_left, pad_right], # Padding for the width dimension
    #  [0, 0]]           # No padding to the channel dimension
    _x = jnp.pad(x,
                 [[0, 0],
                  [pad_bottom, pad_top],
                  [pad_left, pad_right],
                  [0, 0]], mode="constant").astype(jnp.float32) #To ensure all variables are of type float32
    
    # Return the padded array
    return _x

@jit
def rot180(tensor):
    """
    Rotate the input tensor by 180 degrees.

    Args:
        tensor (ndarray): The input tensor to be rotated.

    Returns:
        ndarray: The tensor rotated by 180 degrees.
    """
    
    # Flip the tensor along the first two axes (height and width) to achieve a 180-degree rotation
    flipped_tensor = jnp.flip(tensor, axis=[0, 1])
    
    # Transpose the tensor to reorder the axes
    # The axes [0, 1, 3, 2] correspond to:
    # 0: Height
    # 1: Width
    # 3: Input Channels
    # 2: Output Channels
    rotated_tensor = jnp.transpose(flipped_tensor, axes=[0, 1, 3, 2])
    
    # Return the rotated tensor
    return rotated_tensor

@partial(jit, static_argnums=[2, 3, 4])
def get_same_conv_padding(lhs, rhs, stride_size=1, rhs_dilation=(1, 1),
                          lhs_dilation=(1, 1)):
    padding = "SAME"
    window_strides = (stride_size, stride_size)
    dimension_numbers = ('NHWC', 'HWIO', 'NHWC')
    dnums = lax.conv_dimension_numbers(lhs.shape, rhs.shape, dimension_numbers)
    lhs_perm, rhs_perm, _ = dnums
    rhs_shape = jnp.take(rhs.shape, rhs_perm)[2:]  # type: ignore[index]
    effective_rhs_shape = [core.dilate_dim(k, r) for k, r in
                           zip(rhs_shape, rhs_dilation)]
    padding = lax.padtype_to_pads(
        jnp.take(lhs.shape, lhs_perm)[2:], effective_rhs_shape,
        # type: ignore[index]
        window_strides, padding)
    return padding


@partial(jit, static_argnums=[2, 3, 4])
def get_valid_conv_padding(lhs, rhs, stride_size=1, rhs_dilation=(1, 1),
                           lhs_dilation=(1, 1)):
    padding = "VALID"
    window_strides = (stride_size, stride_size)
    dimension_numbers = ('NHWC', 'HWIO', 'NHWC')
    dnums = lax.conv_dimension_numbers(lhs.shape, rhs.shape, dimension_numbers)
    lhs_perm, rhs_perm, _ = dnums
    rhs_shape = jnp.take(rhs.shape, rhs_perm)[2:]  # type: ignore[index]
    effective_rhs_shape = [core.dilate_dim(k, r) for k, r in
                           zip(rhs_shape, rhs_dilation)]
    padding = lax.padtype_to_pads(
        jnp.take(lhs.shape, lhs_perm)[2:], effective_rhs_shape,
        # type: ignore[index]
        window_strides, padding)
    return padding

def _conv_same_transpose_padding(inputs, output, kernel, stride):
    """
    Calculate the padding for a transpose convolution operation to achieve 'SAME' padding.

    Args:
        inputs (int): The size of the input.

        output (int): The size of the output.

        kernel (int): The size of the convolution kernel.

        stride (int): The stride length of the convolution.

    Returns:
        tuple: The padding for the height and width dimensions.
    """
    pad_len = output - ((stride - 1) * (inputs - 1) + inputs - (kernel - 1))
    if stride > kernel:
        pad_a = kernel - 1
    else:
        pad_a = int(np.ceil(pad_len / 2))
    pad_b = pad_len - pad_a
    return ((pad_a, pad_b), (pad_a, pad_b))

## Better optimized version of conv_same_padding
#@jit
# def _conv_same_transpose_padding(inputs, output, kernel, stride):
#     """
#     Calculate the padding for a transpose convolution operation to achieve 'same' padding.
#
#     Parameters:
#     inputs (int): The size of the input.
#     output (int): The size of the output.
#     kernel (int): The size of the convolution kernel.
#     stride (int): The stride length of the convolution.
#
#     Returns:
#     tuple: The padding for the height and width dimensions.
#     """
#     # Calculate the total padding length required
#     pad_len = output - ((stride - 1) * (inputs - 1) + inputs - (kernel - 1))
#
#     # Determine padding based on stride and kernel size
#     pad_a = jnp.where(stride > kernel, kernel - 1, jnp.ceil(pad_len / 2).astype(int))
#     pad_b = pad_len - pad_a
#
#     # Return the padding for height and width as tuples
#     return ((pad_a, pad_b), (pad_a, pad_b))


def _conv_valid_transpose_padding(inputs, output, kernel, stride):
    """
    Calculate the padding for a transpose convolution operation to achieve 'VALID' padding.

    Args:
        inputs (int): The size of the input.

        output (int): The size of the output.

        kernel (int): The size of the convolution kernel.

        stride (int): The stride length of the convolution.

    Returns:
        tuple: The padding for the height and width dimensions.
    """
    pad_len = output - ((stride - 1) * (inputs - 1) + inputs - (kernel - 1))
    pad_a = kernel - 1
    pad_b = pad_len - pad_a
    return ((pad_a, pad_b), (pad_a, pad_b))

# ## Optimized version version2 for conv_valid_transpose_padding
# @jit
# def _conv_valid_transpose_padding(inputs, output, kernel, stride):
#     """
#     Calculate the padding for a transpose convolution operation to achieve 'valid' padding.
#
#     Parameters:
#     inputs (int): The size of the input.
#     output (int): The size of the output.
#     kernel (int): The size of the convolution kernel.
#     stride (int): The stride length of the convolution.
#
#     Returns:
#     tuple: The padding for the height and width dimensions.
#     """
#     # Calculate the total padding length required
#     pad_len = output - ((stride - 1) * (inputs - 1) + inputs - (kernel - 1))
#
#     # Set the padding value
#     pad_a = kernel - 1
#     pad_b = pad_len - pad_a
#
#     # Ensure values are cast to integers
#     pad_a = jnp.int32(pad_a)
#     pad_b = jnp.int32(pad_b)
#
#     # Return the padding for height and width as tuples
#     return ((pad_a, pad_b), (pad_a, pad_b))


# @jit
# def _deconv_valid_transpose_padding(inputs, output, kernel, stride):
#     """
#     Calculate the padding for a transpose deconvolution operation to achieve 'valid' padding.
#
#     Parameters:
#     inputs (int): The size of the input.
#     output (int): The size of the output.
#     kernel (int): The size of the deconvolution kernel.
#     stride (int): The stride length of the deconvolution.
#
#     Returns:
#     tuple: The padding for the height and width dimensions.
#     """
#     # Calculate the total padding length required
#     pad_len = output - ((stride - 1) * (inputs - 1) + inputs - (kernel - 1))
#
#     # Set the padding value
#     pad_a = output - 1
#     pad_b = pad_len - pad_a
#
#     # Ensure values are cast to integers
#     pad_a = jnp.int32(pad_a)
#     pad_b = jnp.int32(pad_b)
#
#     # Return the padding for height and width as tuples
#     return ((pad_a, pad_b), (pad_a, pad_b))
    
def _deconv_valid_transpose_padding(inputs, output, kernel, stride):
    """
    Calculate the padding for a transpose deconvolution operation to achieve 'VALID' padding.

    Args::
        inputs (int): The size of the input.

        output (int: The size of the output.

        kernel (int): The size of the deconvolution kernel.

        stride (int): The stride length of the deconvolution.

    Returns:
        tuple: The padding for the height and width dimensions.
    """
    pad_len = output - ((stride - 1) * (inputs - 1) + inputs - (kernel - 1))
    pad_a = output - 1
    pad_b = pad_len - pad_a
    return ((pad_a, pad_b), (pad_a, pad_b))

# @jit
# def _deconv_same_transpose_padding(inputs, output, kernel, stride):
#     """
#     Calculate the padding for a transpose deconvolution operation to achieve 'same' padding.
#
#     Parameters:
#     inputs (int): The size of the input.
#     output (int): The size of the output.
#     kernel (int): The size of the deconvolution kernel.
#     stride (int): The stride length of the deconvolution.
#
#     Returns:
#     tuple: The padding for the height and width dimensions.
#     """
#     # Calculate the total padding length required
#     pad_len = output - ((stride - 1) * (inputs - 1) + inputs - (kernel - 1))
#
#     # Determine padding values
#     pad_a = jnp.where(stride >= output - 1, output - 1, jnp.ceil(pad_len / 2)).astype(int)
#     pad_b = pad_len - pad_a
#
#     # Return the padding for height and width as tuples
#     return ((pad_a, pad_b), (pad_a, pad_b))

def _deconv_same_transpose_padding(inputs, output, kernel, stride):
    """
    Calculate the padding for a transpose deconvolution operation to achieve 'SAME' padding.

    Args:
        inputs (int): The size of the input.

        output (int): The size of the output.

        kernel (int): The size of the deconvolution kernel.

        stride (int): The stride length of the deconvolution.

    Returns:
        tuple: The padding for the height and width dimensions.
    """
    pad_len = output - ((stride - 1) * (inputs - 1) + inputs - (kernel - 1))
    if stride >= output - 1:
        pad_a = output - 1
    else:
        pad_a = int(np.ceil(pad_len / 2)) # int(jnp.ceil(pad_len / 2))
    pad_b = pad_len - pad_a
    return ((pad_a, pad_b), (pad_a, pad_b))

@partial(jit, static_argnums=[2, 3, 4])
def deconv2d(inputs, filters, stride_size=1, rhs_dilation=(1, 1),
             padding=((0, 0), (0, 0))):  ## Deconv2D
    dim_numbers = ('NHWC', 'HWIO', 'NHWC')
    out = lax.conv_transpose(inputs,  # lhs = image tensor
                             filters,  # rhs = conv kernel tensor
                             (stride_size, stride_size),  # window strides
                             padding,  # padding mode
                             rhs_dilation,  # rhs/kernel dilation
                             dim_numbers)
    return out


@partial(jit, static_argnums=[2, 3, 4, 5])
def conv2d(inputs, filters, stride_size=1, rhs_dilation=(1, 1),
           lhs_dilation=(1, 1), padding=((0, 0), (0, 0))):  ## Conv2D
    dim_numbers = ('NHWC', 'HWIO', 'NHWC')
    out = lax.conv_general_dilated(inputs,  # lhs = image tensor
                                   filters,  # rhs = conv kernel tensor
                                   (stride_size, stride_size),  # window strides
                                   padding,  # padding mode
                                   lhs_dilation,  # lhs/image dilation
                                   rhs_dilation,  # rhs/kernel dilation
                                   dim_numbers)
    return out

################################################################################
## ngc-learn convolution calculations

# @partial(jit, static_argnums=[2, 3, 4])
# def calc_dK_conv(x, d_out, delta_shape, stride_size=1, padding=((0, 0), (0, 0))):
#     _x = x
#     deX, deY = delta_shape
#     if deX > 0:
#         ## apply a pre-computation trimming step ("negative padding")
#         _x = x[:, 0:x.shape[1]-deX, 0:x.shape[2]-deY, :]
#     return _calc_dK_conv(_x, d_out, stride_size=stride_size, padding=padding)

@partial(jit, static_argnums=[2, 3, 4])
def calc_dK_conv(x, d_out, delta_shape, stride_size=1, padding=((0, 0), (0, 0))):
    """
    Calculate the gradient with respect to the kernel for a convolution operation.
    
    Args:
        x (ndarray): The input array.

        d_out (ndarray): The gradient with respect to the output.

        delta_shape (tuple): The shape difference (deX, deY) between the input and output.

        stride_size (int): The stride size for the convolution. Defaults to 1.

        padding (tuple): Padding to apply to the input. Defaults to ((0, 0), (0, 0)).
    
    Returns:
        ndarray: The gradient with respect to the kernel.
    """
    deX, deY = delta_shape

    # Apply a pre-computation trimming step ("negative padding") if needed

    #_x = jnp.where(deX > 0, x[:, 0:x.shape[1]-deX, 0:x.shape[2]-deY, :], x)
    # _deX = jnp.maximum(deX, 0).astype(jnp.int32)
    # _deY = jnp.maximum(deY, 0).astype(jnp.int32)
    _x = x[:, 0:x.shape[1]-deX, 0:x.shape[2]-deY, :]

    # Calculate the gradient with respect to the kernel
    return _calc_dK_conv(_x, d_out, stride_size=stride_size, padding=padding)

@partial(jit, static_argnums=[2, 3])
def _calc_dK_conv(x, d_out, stride_size=1, padding=((0, 0), (0, 0))):
    xT = jnp.transpose(x, axes=[3, 1, 2, 0])
    d_out_T = jnp.transpose(d_out, axes=[1, 2, 0, 3])
    ## original conv2d
    dW = conv2d(inputs=xT, filters=d_out_T, stride_size=1, padding=padding,
                rhs_dilation=(stride_size, stride_size)).astype(jnp.float32)
    return jnp.transpose(dW, axes=[1, 2, 0, 3])

################################################################################
# input update computation
@partial(jit, static_argnums=[2, 3, 4])
def calc_dX_conv(K, d_out, delta_shape, stride_size=1, anti_padding=None):
    deX, deY = delta_shape
    # if abs(deX) > 0 and stride_size > 1:
    #     return _calc_dX_subset(K, d_out, (abs(deX),abs(deY)), stride_size=stride_size,
    #                            anti_padding=anti_padding)
    dx = _calc_dX_conv(K, d_out, stride_size=stride_size, anti_padding=anti_padding)
    return dx

@partial(jit, static_argnums=[2, 3])
def _calc_dX_conv(K, d_out, stride_size=1, anti_padding=None):
    w_size = K.shape[0]
    K_T = rot180(K)  # Assuming rot180 is defined elsewhere.
    _pad = w_size - 1
    return deconv2d(d_out, filters=K_T, stride_size=stride_size,
                    padding=anti_padding).astype(jnp.float32)

################################################################################
## ngc-learn deconvolution calculations

# @partial(jit, static_argnums=[2, 3, 4, 5])
# def calc_dK_deconv(x, d_out, delta_shape, stride_size=1, out_size =2, padding="SAME"):
#     _x = x
#     deX, deY = delta_shape
#     if deX > 0:
#         ## apply a pre-computation trimming step ("negative padding")
#         _x = x[:, 0:x.shape[1]-deX, 0:x.shape[2]-deY, :]
#     return _calc_dK_deconv(_x, d_out, stride_size=stride_size, out_size = out_size)

@partial(jit, static_argnums=[2, 3, 4, 5])
def calc_dK_deconv(x, d_out, delta_shape, stride_size=1, out_size=2, padding="SAME"):
    """
    Calculate the gradient with respect to the kernel for a deconvolution operation.
    
    Args:
        x (ndarray): The input array.

        d_out (ndarray): The gradient with respect to the output.

        delta_shape (tuple): The shape difference (deX, deY) between the input and output.

        stride_size (int): The stride size for the deconvolution. Defaults to 1.

        out_size (int): The output size for the deconvolution.

        padding (str): Padding to apply to the input. Defaults to "SAME".
    
    Returns:
        ndarray: The gradient with respect to the kernel.
    """
    deX, deY = delta_shape

    # Apply a pre-computation trimming step ("negative padding") if needed
    #_x = jnp.where(deX > 0, x[:, :x.shape[1]-deX, :x.shape[2]-deY, :], x)
    _x = x[:, :x.shape[1]-deX, :x.shape[2]-deY, :]

    # Calculate the gradient with respect to the kernel
    return _calc_dK_deconv(_x, d_out, stride_size=stride_size, out_size=out_size, padding=padding)


@partial(jit, static_argnums=[2, 3, 4])
def _calc_dK_deconv(x, d_out, stride_size=1, out_size=2, padding="SAME"):
    xT = jnp.transpose(x, axes=[3, 1, 2, 0])
    d_out_T = jnp.transpose(d_out, axes=[1, 2, 0, 3])
    if padding == "VALID":
        pad_args = _deconv_valid_transpose_padding(xT.shape[1], out_size, d_out_T.shape[1], stride_size)
    elif padding == "SAME":
        pad_args = _deconv_same_transpose_padding(xT.shape[1], out_size, d_out_T.shape[1], stride_size)
    dW = deconv2d(inputs=xT, filters=d_out_T, stride_size=stride_size,
                  padding=pad_args)
    dW = jnp.transpose(dW, axes=[1, 2, 0, 3])
    return dW
################################################################################
# input update computation
@partial(jit, static_argnums=[2, 3, 4])
def calc_dX_deconv(K, d_out, delta_shape, stride_size=1, padding=((0, 0), (0, 0))):
    """
    Wrapper function to calculate the gradient with respect to the input (dX)
    from the gradient with respect to the output (d_out) using the kernel (K).
    This version takes into account the shape difference (delta_shape) between
    the input and the output of the convolution.

    Args:
        K (ndarray): The convolution kernel.

        d_out (ndarray): The gradient with respect to the output of the convolution.

        delta_shape (tuple): The shape difference (deX, deY) between the input and output.

        stride_size (int): The stride size for the deconvolution. Defaults to 1.

        padding (tuple): Padding to apply to the input. Defaults to ((0, 0), (0, 0)).

    Returns:
        dx (ndarray): The gradient with respect to the input.
    """
    
    # Extract the shape difference in the x and y dimensions
    deX, deY = delta_shape

    # Conditional logic to handle cases where delta_shape is non-zero and stride_size is greater than 1
    # if abs(deX) > 0 and stride_size > 1:
    #     return _calc_dX_subset(K, d_out, (abs(deX), abs(deY)), stride_size=stride_size, padding=padding)

    # Call the _calc_dX_deconv function to perform the deconvolution
    dx = _calc_dX_deconv(K, d_out, stride_size=stride_size, padding=padding)
    
    # Return the gradient with respect to the input
    return dx

@partial(jit, static_argnums=[2, 3])
def _calc_dX_deconv(K, d_out, stride_size=1, padding=((0, 0), (0, 0))):
    """
    Perform a deconvolution to get the gradient with respect to the input (dX)
    from the gradient with respect to the output (d_out) using the kernel (K).

    Args:
        K (ndarray): The convolution kernel.

        d_out (ndarray): The gradient with respect to the output of the convolution.

        stride_size (int): The stride size for the deconvolution. Defaults to 1.

        padding (tuple): Padding to apply to the input. Defaults to ((0, 0), (0, 0)).

    Returns:
        dx (ndarray): The gradient with respect to the input.
    """

    # The size of the kernel
    w_size = K.shape[0]

    # Rotate the kernel by 180 degrees
    K_T = rot180(K)  # Equivalent to jnp.transpose(K, axes=[1, 0, 3, 2])
    
    # Padding size for the deconvolution, derived from the kernel size
    _pad = w_size - 1

    # Perform the deconvolution using conv2d with the rotated kernel
    dx = conv2d(d_out,
                filters=K_T,
                stride_size=stride_size,
                padding=padding)
    
    # Return the gradient with respect to the input
    return dx
################################################################################
