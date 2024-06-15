import numpy as np
from jax import jit, numpy as jnp, random, nn, lax
from functools import partial
from jax._src import core

@partial(jit, static_argnums=[1])
def _pad(x, padding):
    """
    Jit-i-fied padding co-routine
    """
    pad_bottom, pad_top, pad_left, pad_right = padding
    _x = jnp.pad(x,
                 [[0, 0],
                  [pad_bottom, pad_top],
                  [pad_left, pad_right],
                  [0, 0]], mode="constant").astype(jnp.float32)
    return _x

@jit
def rot180(M):
    """
    Rotate input M by 180 degrees
    """
    return jnp.transpose(jnp.flip(M, axis=[0, 1]), axes=[0, 1, 3, 2])


@partial(jit, static_argnums=[2, 3, 4])
def get_same_conv_padding(lhs, rhs, stride_size=1, rhs_dilation=(1, 1),
                          lhs_dilation=(1, 1)):
    padding = "SAME"
    window_strides = (stride_size, stride_size)
    dimension_numbers = ('NHWC', 'HWIO', 'NHWC')
    dnums = lax.conv_dimension_numbers(lhs.shape, rhs.shape, dimension_numbers)
    lhs_perm, rhs_perm, _ = dnums
    rhs_shape = np.take(rhs.shape, rhs_perm)[2:]  # type: ignore[index]
    effective_rhs_shape = [core.dilate_dim(k, r) for k, r in
                           zip(rhs_shape, rhs_dilation)]
    padding = lax.padtype_to_pads(
        np.take(lhs.shape, lhs_perm)[2:], effective_rhs_shape,
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
    rhs_shape = np.take(rhs.shape, rhs_perm)[2:]  # type: ignore[index]
    effective_rhs_shape = [core.dilate_dim(k, r) for k, r in
                           zip(rhs_shape, rhs_dilation)]
    padding = lax.padtype_to_pads(
        np.take(lhs.shape, lhs_perm)[2:], effective_rhs_shape,
        # type: ignore[index]
        window_strides, padding)
    return padding

def _conv_same_transpose_padding(inputs, output, kernel, stride):
    pad_len = output - ((stride - 1) * (inputs - 1) + inputs - (kernel - 1))
    if stride > kernel:
        pad_a = kernel - 1
    else:
        pad_a = int(np.ceil(pad_len / 2))
    print("pad_len = ", pad_len)
    pad_b = pad_len - pad_a
    return ((pad_a, pad_b), (pad_a, pad_b))


def _conv_valid_transpose_padding(inputs, output, kernel, stride):
    print("k : ", kernel, "  s : ", stride, " input : ", inputs)
    pad_len = output - ((stride - 1) * (inputs - 1) + inputs - (kernel - 1))
    pad_a = kernel - 1
    print("pad_len = ", pad_len)
    pad_b = pad_len - pad_a
    return ((pad_a, pad_b), (pad_a, pad_b))


def _deconv_valid_transpose_padding(inputs, output, kernel, stride):
    print(inputs, output, kernel, stride)
    pad_len = output - ((stride - 1) * (inputs - 1) + inputs - (kernel - 1))
    pad_a = output - 1
    print("pad_len = ", pad_len)
    pad_b = pad_len - pad_a
    return ((pad_a, pad_b), (pad_a, pad_b))


def _deconv_same_transpose_padding(inputs, output, kernel, stride):
    print(inputs, output, kernel, stride)
    pad_len = output - ((stride - 1) * (inputs - 1) + inputs - (kernel - 1))
    if stride >= output - 1:
        pad_a = output - 1
    else:
        pad_a = int(np.ceil(pad_len / 2))
    print("pad_len = ", pad_len)
    pad_b = pad_len - pad_a
    return ((pad_a, pad_b), (pad_a, pad_b))

@partial(jit, static_argnums=[2, 3, 4])
def deconv2d(inputs, filters, stride_size=1, rhs_dilation=(1, 1),
             padding=((0, 0), (0, 0))):  ## Deconv2D
    dim_numbers = ('NHWC', 'HWIO', 'NHWC')
    print("---- in deconv ----")
    print("padding : ", padding)
    print(inputs.shape, filters.shape)
    # padding = "SAME"
    out = lax.conv_transpose(inputs,  # lhs = image tensor
                             filters,  # rhs = conv kernel tensor
                             (stride_size, stride_size),  # window strides
                             padding,  # padding mode
                             rhs_dilation,  # rhs/kernel dilation
                             dim_numbers)
    print(out.shape)
    # print(out[0,:,:,0])
    print("----------------------")
    return out


@partial(jit, static_argnums=[2, 3, 4, 5])
def conv2d(inputs, filters, stride_size=1, rhs_dilation=(1, 1),
           lhs_dilation=(1, 1), padding=((0, 0), (0, 0))):  ## Conv2D
    # padding = ((0,0),(0,0)) #"VALID"
    # padding = "SAME"
    dim_numbers = ('NHWC', 'HWIO', 'NHWC')
    print("---- in conv ----")
    print("padding : ", padding)
    print(inputs.shape, filters.shape)
    out = lax.conv_general_dilated(inputs,  # lhs = image tensor
                                   filters,  # rhs = conv kernel tensor
                                   (stride_size, stride_size),  # window strides
                                   padding,  # padding mode
                                   lhs_dilation,  # lhs/image dilation
                                   rhs_dilation,  # rhs/kernel dilation
                                   dim_numbers)
    print(out.shape)
    print("----------------------")
    return out
