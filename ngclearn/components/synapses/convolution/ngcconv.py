"""
Calculation toolbox that drives conv/deconv operations in the ngc-learn
convolution components sub-branch; this contains routines/co-routines
for `ngclearn.components.synapses.convolution`.
"""
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
def rot180(tensor):
    """
    Rotate input M by 180 degrees
    """
    return jnp.transpose(jnp.flip(tensor, axis=[0, 1]), axes=[0, 1, 3, 2])


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
    pad_len = output - ((stride - 1) * (inputs - 1) + inputs - (kernel - 1))
    if stride > kernel:
        pad_a = kernel - 1
    else:
        pad_a = int(jnp.ceil(pad_len / 2)) # int(jnp.ceil(pad_len / 2))
    pad_b = pad_len - pad_a
    return ((pad_a, pad_b), (pad_a, pad_b))


def _conv_valid_transpose_padding(inputs, output, kernel, stride):
    pad_len = output - ((stride - 1) * (inputs - 1) + inputs - (kernel - 1))
    pad_a = kernel - 1
    pad_b = pad_len - pad_a
    return ((pad_a, pad_b), (pad_a, pad_b))


def _deconv_valid_transpose_padding(inputs, output, kernel, stride):
    pad_len = output - ((stride - 1) * (inputs - 1) + inputs - (kernel - 1))
    pad_a = output - 1
    pad_b = pad_len - pad_a
    return ((pad_a, pad_b), (pad_a, pad_b))


def _deconv_same_transpose_padding(inputs, output, kernel, stride):
    pad_len = output - ((stride - 1) * (inputs - 1) + inputs - (kernel - 1))
    if stride >= output - 1:
        pad_a = output - 1
    else:
        pad_a = int(jnp.ceil(pad_len / 2)) # int(jnp.ceil(pad_len / 2))
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

@partial(jit, static_argnums=[2, 3, 4])
def calc_dK_conv(x, d_out, delta_shape, stride_size=1, padding=((0, 0), (0, 0))):
    _x = x
    deX, deY = delta_shape
    if deX > 0:
        ## apply a pre-computation trimming step ("negative padding")
        _x = x[:, 0:x.shape[1]-deX, 0:x.shape[2]-deY, :]
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

@partial(jit, static_argnums=[2, 3, 4, 5])
def calc_dK_deconv(x, d_out, delta_shape, stride_size=1, out_size =2, padding="SAME"):
    _x = x
    deX, deY = delta_shape
    if deX > 0:
        ## apply a pre-computation trimming step ("negative padding")
        _x = x[:, 0:x.shape[1]-deX, 0:x.shape[2]-deY, :]
    return _calc_dK_deconv(_x, d_out, stride_size=stride_size, out_size = out_size)

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
    deX, deY = delta_shape
    # if abs(deX) > 0 and stride_size > 1:
    #     return _calc_dX_subset(K, d_out, (abs(deX), abs(deY)), stride_size=stride_size, padding = padding)
    dx = _calc_dX_deconv(K, d_out, stride_size=stride_size, padding=padding)
    return dx

@partial(jit, static_argnums=[2,3])
def _calc_dX_deconv(K, d_out, stride_size=1, padding=((0, 0), (0, 0))):
    ## deconvolution is done to get "through" a convolution backwards
    w_size = K.shape[0]
    K_T = rot180(K) #jnp.transpose(K, axes=[1,0,3,2])
    _pad = w_size - 1
    dx = conv2d(d_out,
                filters=K_T,
                stride_size=stride_size,
                padding=padding)
    return dx
################################################################################