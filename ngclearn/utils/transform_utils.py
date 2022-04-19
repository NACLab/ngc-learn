"""
Copyright (C) 2021 Alexander G. Ororbia II - All Rights Reserved
You may use, distribute and modify this code under the
terms of the GNU LGPL-3.0-or-later license.

You should have received a copy of the XYZ license with
this file. If not, please write to: ago@cs.rit.edu , or visit:
https://www.gnu.org/licenses/lgpl-3.0.en.html
"""

"""
A mathematical transformation utilities function file.

@author: Alexander Ororbia
"""
import tensorflow as tf
import numpy as np

def binarize(data, threshold=0.5):
    """Converts the vector *data* to its binary equivalent"""
    return tf.cast(tf.greater_equal(data, threshold),dtype=tf.float32)

def convert_to_spikes(x_data, max_spike_rate, dt, sp_div=4.0):
    """
    Converts a vector *x_data* to its approximate Poisson spike equivalent

    max_spike_rate: firing rate (in Hertz)
    dt: integraton time constant (in milliseconds or ms)
    sp_div: to denominator to convert input data values to a firing frequency
    Return:
    """
    noise_eps = tf.random.uniform(x_data.shape)
    # Spike train via: https://www.frontiersin.org/articles/10.3389/fncom.2015.00099/full
    x_freq = (x_data)/sp_div # divide to convert un-scaled data to firing rates
    thresh = tf.cast(x_freq, dtype=tf.float32) * ((1.0 * dt)/1000.0) # between 0 and 63.75 Hz
    # Spike train via Ororbia-style
    #thresh = tf.cast(x_data, dtype=tf.float32) * ((max_spike_rate * dt)/1000.0) # between 0 and 64 Hz if max_spike_rate = 64.0
    x_sp = tf.math.less(noise_eps, thresh)
    x_sp = tf.cast(x_sp, dtype=tf.float32)
    return x_sp

def filter(x_t, x_f, dt, a, filter_type="var_trace"):
    """ Applies a filter to data *x_t* """
    if filter_type == "var_trace":
        # apply variable trace filters z_l(t) = (alpha * z_l(t))*(1−s`(t)) +s_l(t)
        x_f_ = tf.add((x_f * a) * (-x_t + 1.0), x_t)
    else:
        #f(t) = (1 / tau) * exp(-t / tau)
        tau = 5.0
        a_f = np.exp(-dt/tau) * (dt/tau)
        x_f_ = x_f * (1.0 - a_f) + x_t * a_f
        # apply low-pass filters -- y(k) = a * y(k-1) + (1-a) * x(k)  where a = exp (-T/tau)
        # R_m = 1.0 # 1 kOhm
        # C_m = 10.0 # 10 pF
        # a_f = dt/(R_m * C_m + dt)
        # x_f_ = x_f * (1.0 - a_f) + x_t * a_f
    return x_f_

def binary_flip(x_b):
    return (-x_b + 1.0)

def decide_fun(fun_type):
    """
    A selector function that generates a physical activation function and its
    first-order (element-wise) derivative funciton given a description *fun_type*
    """
    fx = None
    d_fx = None
    if fun_type == "binary_flip":
        fx = binary_flip
        d_fx = d_identity
    elif "bkwta" in fun_type:
        # n_winners = float(fun_type[fun_type.index("(")+1:fun_type.rindex(")")])
        # # must generate a custom wta function here inside this function creator
        # def tmp_bkwta(x, K=n_winners): # returns only binary code of K top winner nodes
        #     values, indices = tf.math.top_k(x, k=K, sorted=False) # Note: we do not care to sort the indices
        #     kth = tf.expand_dims(tf.reduce_min(values,axis=1),axis=1) # must do comparison per sample in potential mini-batch
        #     topK = tf.cast(tf.greater_equal(x, kth), dtype=tf.float32) # cast booleans to floats
        #     return topK
        # return custom-set WTA function
        fx = bkwta
        d_fx = d_identity
    elif fun_type == "tanh":
        fx = tf.nn.tanh
        d_fx = d_tanh
    elif fun_type == "sign":
        fx = tf.math.sign
        d_fx = d_identity
    elif fun_type == "clip_fx":
        fx = clip_fx
        d_fx = d_identity
    elif fun_type == "gte":
        fx = gte
        d_fx = d_identity
    elif fun_type == "ltanh":
        fx = ltanh
        d_fx = d_ltanh
    elif fun_type == "elu":
        fx = tf.nn.elu
        d_fx = d_identity
    elif fun_type == "erf":
        fx = tf.math.erf
        d_fx = d_identity
    elif fun_type == "lrelu":
        fx = tf.nn.leaky_relu
        d_fx = d_identity
    elif fun_type == "relu":
        fx = tf.nn.relu
        d_fx = d_relu
    elif fun_type == "softplus":
        fx = tf.math.softplus
        d_fx = d_softplus
    elif fun_type == "relu6":
        fx = tf.nn.relu6
        d_fx = d_relu6
    elif fun_type == "sigmoid":
        fx = tf.nn.sigmoid
        d_fx = d_sigmoid
    elif fun_type == "bkwta":
        fx = kwta
        d_fx = d_identity
    elif fun_type == "kwta":
        fx = kwta
        d_fx = bkwta #d_identity
    elif fun_type == "softmax":
        fx = softmax
        d_fx = tf.identity
    else:
        fx = tf.identity
        d_fx = d_identity
    return fx, d_fx

def identity(z):
    return z

def d_identity(x):
    return x * 0 + 1.0

def gte(x, val=0.0):
    return tf.cast(tf.greater_equal(x, val),dtype=tf.float32)

def elu(z,alpha=1.0):
    return z if z >= 0 else (tf.math.exp(z) - 1.0) * alpha

def d_elu(z,alpha=1.0):
	return 1 if z > 0 else tf.math.exp(z) * alpha

def d_sigmoid(x):
    sigm_x = tf.nn.sigmoid(x)
    return (-sigm_x + 1.0) * sigm_x

def d_tanh(x):
    tanh_x = tf.nn.tanh(x)
    return -(tanh_x * tanh_x) + 1.0

def d_relu(x):
    # df/dx = 1 if x >= 0 else 0
    val = tf.math.greater_equal(x, 0.0)
    return tf.cast(val,dtype=tf.float32) # sign(max(0,x))

def d_relu6(x):
    # df/dx = 1 if 0<x<6 else 0
    # I_x = (z >= a_min) *@ (z <= b_max) //create an indicator function  a = 0 b = 6
    Ix1 = tf.cast(tf.math.greater_equal(x, 0.0),dtype=tf.float32)
    Ix2 = tf.cast(tf.math.less_equal(x, 6.0),dtype=tf.float32)
    Ix = Ix1 * Ix2
    return Ix

def d_softplus(x):
    return tf.nn.sigmoid(x) # d/dx of softplus = logistic sigmoid

def softmax(x, tau=0.0):
    """
        Softmax function with overflow control built in directly. Contains optional
        temperature parameter to control sharpness (tau > 1 softens probs, < 1 sharpens --> 0 yields point-mass)
    """
    if tau > 0.0:
        x = x / tau
    max_x = tf.expand_dims( tf.reduce_max(x, axis=1), axis=1)
    exp_x = tf.exp(tf.subtract(x, max_x))
    return exp_x / tf.expand_dims( tf.reduce_sum(exp_x, axis=1), axis=1)

def mellowmax(x, omega=1.0,axis=1):
    n = x.shape[axis] * 1.0
    #(F.logsumexp(omega * values, axis=axis) - np.log(n)) / omega
    return ( tf.reduce_logsumexp(x * omega, axis=axis, keepdims=True) - tf.math.log(n) ) / omega

def ltanh(z):
    a = 1.7159
    b = 2.0/3.0
    z_scale = z * b
    z_scale = tf.clip_by_value(z_scale, -50.0, 50.0) #-85.0, 85.0)
    neg_exp = tf.exp(-z_scale)
    pos_exp = tf.exp(z_scale)
    denom = tf.add(pos_exp, neg_exp)
    numer = tf.subtract(pos_exp, neg_exp)
    return tf.math.divide(numer, denom) * a

def d_ltanh(z):
    a = 1.7159
    b = 2.0/3.0
    z_scale = z * b
    z_scale = tf.clip_by_value(z_scale, -50.0, 50.0) #-85.0, 85.0)
    neg_exp = tf.exp(-z_scale)
    pos_exp = tf.exp(z_scale)
    denom = tf.add(pos_exp, neg_exp)
    dx = tf.math.divide((4.0 * a * b), denom * denom)
    return dx

def clip_fx(x):
    return tf.clip_by_value(x, 0.0, 1.0)

def scale_feat(x, a=-1.0, b=1.0):
    max_x = tf.reduce_max(x,axis=1,keepdims=True)
    min_x = tf.reduce_min(x,axis=1,keepdims=True)
    x_prime = a + ( ( (x - min_x) * (b - a) )/(max_x - min_x) )
    return tf.cast(x_prime, dtype=tf.float32)

def kwta(x, K=50): #5 10 15 #K=50):
    """
        k-winners-take-all competitive activation function
    """
    values, indices = tf.math.top_k(x, k=K, sorted=False) # Note: we do not care to sort the indices
    kth = tf.expand_dims(tf.reduce_min(values,axis=1),axis=1) # must do comparison per sample in potential mini-batch
    topK = tf.cast(tf.greater_equal(x, kth), dtype=tf.float32) # cast booleans to floats
    return topK * x

def bkwta(x, K=10): # returns only binary code of K top winner nodes
    """
        Binarized k-winners-take-all competitive activation function
    """
    values, indices = tf.math.top_k(x, k=K, sorted=False) # Note: we do not care to sort the indices
    kth = tf.expand_dims(tf.reduce_min(values,axis=1),axis=1) # must do comparison per sample in potential mini-batch
    topK = tf.cast(tf.greater_equal(x, kth), dtype=tf.float32) # cast booleans to floats
    return topK

def mish(x):
    return x * tf.nn.tanh(tf.math.softplus(x))

def shrink(a,b):
    return tf.math.sign(a) * tf.maximum(tf.math.abs(a) - b, 0.0)

def drop_out(input, rate=0.0, seed=69):
    """
        Custom drop-out function -- returns output as well as binary mask
    """
    mask = tf.math.less_equal( tf.random.uniform(shape=(input.shape[0],input.shape[1]), minval=0.0, maxval=1.0, dtype=tf.float32, seed=seed),(1.0 - rate))
    mask = tf.cast(mask, tf.float32) * (1.0 / (1.0 - rate))
    output = input * mask
    return output, mask

def create_block_bin_matrix(shape, n_ones_per_row):
    nrows, ncols = shape
    bin_mat = None
    ones = tf.ones([1,n_ones_per_row])
    for r in range(nrows):
        row_r = None
        if r == (nrows - 1):
            row_r = tf.concat([tf.zeros([1,ncols - n_ones_per_row]), ones],axis=1)
        elif r == 0:
            row_r = tf.concat([ones, tf.zeros([1,ncols - n_ones_per_row])],axis=1)
        else:
            left = tf.zeros([1,n_ones_per_row * r])
            right = tf.zeros([1,ncols - (left.shape[1] + ones.shape[1])])
            row_r = tf.concat([left, ones, right],axis=1)
        if r > 0:
            bin_mat = tf.concat([bin_mat,row_r],axis=0)
        else:
            bin_mat = row_r
    return bin_mat

def init_weights(kernel, shape, seed):
    """
        Randomly generates/initializes a matrix/vector according to a kernel pattern
    """
    init_type = kernel[0]
    params = None
    if init_type == "anti_diagonal":
        n_cols = shape[1]
        factor = kernel[1]
        I = tf.eye(n_cols) # create diagonal I
        AI = (1.0 - I) * factor # create anti-diagonal AI
        params = AI
    elif init_type == "inhibit_matrix": # special matrix for current inhibition
        #M = (I - (1.0 - I)) * (f/2.0)
        factor = kernel[1]
        n_cols = shape[1]
        I = tf.eye(n_cols) # create diagonal I
        M = (I - (1.0 - I)) * (factor/2.0)
        params = M
    elif init_type == "block_bin":
        n_ones_per_row = kernel[1]
        accept_coeff = kernel[2] # 0.5
        reject_coeff = kernel[3] # 100.0
        params = create_block_bin_matrix(shape, n_ones_per_row)
        complement = tf.cast(tf.math.equal(params, 0.0),dtype=tf.float32) * reject_coeff
        params = tf.cast(params,dtype=tf.float32) * accept_coeff
        params = params + complement
    elif init_type == "he_uniform":
        initializer = tf.compat.v1.keras.initializers.he_uniform()
        params = initializer(shape) #, seed=seed )
    elif init_type == "he_normal":
        initializer = tf.compat.v1.keras.initializers.he_normal()
        params = initializer(shape) #, seed=seed )
    elif init_type is "classic_glorot":
        N = (shape[0] + shape[1]) * 1.0
        bound = 4.0 * np.sqrt(6.0/N)
        params = tf.random.uniform(shape, minval=-bound, maxval=bound, seed=seed)
    elif init_type is "glorot_normal":
        initializer = tf.compat.v1.keras.initializers.glorot_normal()
        params = initializer(shape) #, seed=seed )
    elif init_type is "glorot_uniform":
        initializer = tf.compat.v1.keras.initializers.glorot_uniform()
        params = initializer(shape) #, seed=seed )
    elif init_type is "orthogonal":
        stddev = kernel[1]
        initializer = tf.compat.v1.keras.initializers.orthogonal(gain=stddev)
        params = initializer(shape)
    elif init_type is "truncated_normal" or init_type == "truncated_gaussian" :
        stddev = kernel[1]
        params = tf.random.truncated_normal(shape, stddev=stddev, seed=seed)
    elif init_type == "normal" or init_type == "gaussian" :
        stddev = kernel[1]
        params = tf.random.normal(shape, stddev=stddev, seed=seed)
    elif init_type is "alex_uniform": #
        k = 1.0 / (shape[0] * 1.0) # 1/in_features
        bound = np.sqrt(k)
        params = tf.random.uniform(shape, minval=-bound, maxval=bound, seed=seed)
    else: # zeros
        params = tf.zeros(shape)
    params = tf.cast(params,dtype=tf.float32)
    return params

def create_competiion_matrix(z_dim, lat_type, beta_scale, alpha_scale, n_group, band):
    """
        This function creates a particular matrix to simulate competition via self-excitatory
        and inhibitory synaptic signals.
    """
    V_l = None
    if lat_type == "band":
        # nearby-neighbor band inhibition
        V_l = tf.ones([z_dim,z_dim])
        diag = tf.eye(z_dim)
        V_inh = tf.linalg.band_part(V_l, band, band) * (1.0 - diag) * beta_scale
        V_l = V_inh + diag * alpha_scale
    elif lat_type == "lkwta":
        diag = tf.eye(z_dim)
        V_l = None
        g_shift = 0
        while (z_dim - (n_group + g_shift)) >= 0:
            if g_shift > 0:
                left = tf.zeros([1,g_shift])
                middle = tf.ones([1,n_group])
                right = tf.zeros([1,z_dim - (n_group + g_shift)])
                slice = tf.concat([left,middle,right],axis=1)
                for n in range(n_group):
                    V_l = tf.concat([V_l,slice],axis=0)
            else:
                middle = tf.ones([1,n_group])
                right = tf.zeros([1,z_dim - n_group])
                slice = tf.concat([middle,right],axis=1)
                for n in range(n_group):
                    if V_l is not None:
                        V_l = tf.concat([V_l,slice],axis=0)
                    else:
                        V_l = slice
            g_shift += n_group
        V_l = V_l * (1.0 - diag) * beta_scale + diag * alpha_scale
    return V_l

def calc_modulatory_factor(W):
    """
        Calculate modulatory matrix W_M for W
    """
    iL = tf.reduce_sum(tf.math.abs(W),axis=0,keepdims=True)
    iL = tf.math.minimum(iL / tf.reduce_max(iL) * 2, 1.0)
    W_M_l = (W * 0) + iL
    return W_M_l

def create_mask_matrix(n_col_m, nrow, ncol):
    mask = None
    for c in range(ncol):
        if c < n_col_m:
            col = tf.ones([nrow,1])
        else:
            col = tf.zeros([nrow,1])
        if c > 0:
            mask = tf.concat([mask, col],axis=1)
        else:
            mask = col
    return mask
