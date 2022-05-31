"""
A mathematical transformation utilities function file. This file contains
activation functions and other relevant data transformation tools/utilities.
"""
import tensorflow as tf
import numpy as np
from ngclearn.utils.stat_utils import sample_bernoulli

def decide_fun(fun_type):
    """
    A selector function that generates a physical activation function and its
    first-order (element-wise) derivative funciton given a description *fun_type*.
    Note that some functions do not come with a proper derivative (and thus set to
    the identity function derivative -- see list below).

    | Currently supported functions (for given fun_type) include:
    |   * "tanh" - hyperbolic tangent
    |   * "ltanh" - LeCun-style hyperbolic tangent
    |   * "sigmoid" - logistic link function
    |   * "kwta" - K-winners-take-all
    |   * "softmax" - the softmax function (derivative not generated)
    |   * "identity" - the identity function
    |   * "relu" - rectified linear unit
    |   * "lrelu" - leaky rectified linear unit
    |   * "softplus" - the softplus function
    |   * "relu6" - the relu but upper bounded/capped at 6.0
    |   * "elu" - exponential linear unit
    |   * "erf" - the error function (derivative not generated)
    |   * "binary_flip" - bit-flipping function (derivative not generated)
    |   * "bkwta" - binary K-winners-take-all (derivative not generated)
    |   * "sign" - signum (derivative not generated)
    |   * "clip_fx" - clipping function (derivative not generated)
    |   * "heaviside" - Heaviside function  (derivative not generated)
    |   * "bernoulli" - the Bernoulli sampling function  (derivative not generated)

    Args:
        fun_type: a string stating the name of activation function and its 1st
            elementwise derivative to generate

    Return:
        (fx, d_fx), where fx is the physical activation function and d_fx its derivative
    """
    fx = None
    d_fx = None
    if fun_type == "binary_flip":
        fx = binary_flip
        d_fx = d_identity
    elif "bkwta" in fun_type:
        fx = bkwta
        d_fx = d_identity
    elif "kwta" in fun_type:
        fx = kwta
        d_fx = bkwta
    elif fun_type == "tanh":
        fx = tanh #tf.nn.tanh
        d_fx = d_tanh
    elif fun_type == "sign":
        fx = sign #tf.math.sign
        d_fx = d_identity
    elif fun_type == "clip_fx":
        fx = clip_fx
        d_fx = d_identity
    elif fun_type == "heaviside":
        fx = gte
        d_fx = d_identity
    elif fun_type == "ltanh":
        fx = ltanh
        d_fx = d_ltanh
    elif fun_type == "elu":
        fx = elu
        d_fx = d_elu
    elif fun_type == "erf":
        fx = erf
        d_fx = d_identity
    elif fun_type == "lrelu":
        fx = lrelu
        d_fx = d_identity
    elif fun_type == "relu":
        fx = relu
        d_fx = d_relu
    elif fun_type == "softplus":
        fx = softplus
        d_fx = d_softplus
    elif fun_type == "relu6":
        fx = relu6
        d_fx = d_relu6
    elif fun_type == "sigmoid":
        fx = sigmoid
        d_fx = d_sigmoid
    elif fun_type == "softmax":
        fx = softmax
        d_fx = tf.identity
    elif fun_type == "bernoulli":
        fx = bernoulli
        d_fx = tf.identity
    elif fun_type == "binarize":
        fx = binarize
        d_fx = tf.identity
    else:
        fx = tf.identity
        d_fx = d_identity
    return fx, d_fx

def init_weights(kernel, shape, seed):
    """
    Randomly generates/initializes a matrix/vector according to a kernel pattern.

    | Currently supported/tested patterns include:
    |   * "he_uniform"
    |   * "he_normal"
    |   * "classic_glorot"
    |   * "glorot_normal"
    |   * "glorot_uniform"
    |   * "orthogonal"
    |   * "truncated_gaussian" (alternative: "truncated_normal")
    |   * "gaussian" (alternative: "normal")
    |   * "uniform"

    Args:
        kernel: a tuple denoting the pattern by which a matrix is initialized
            Note that the first item of *kernel* MUST contain a string specifying
            the initlialization pattern/scheme to use. Other elements, for tuples
            of length > 1 can contain pattern-specific hyper-paramters.

        shape: a 2-tuple specifying (N x M), a matrix of N rows by M columns

        seed: value to control determinism in initializer

    Returns:
        an (N x M) matrix randomly initialized to a chosen scheme
    """
    init_type = kernel[0]
    params = None
    if init_type == "diagonal":
        n_cols = shape[1]
        factor = kernel[1]
        I = tf.eye(n_cols) # create diagonal I
        params = I
    elif init_type == "anti_diagonal":
        n_cols = shape[1]
        factor = kernel[1]
        I = tf.eye(n_cols) # create diagonal I
        AI = (1.0 - I) * factor # create anti-diagonal AI
        params = AI
    elif init_type == "inhibit_matrix": # special matrix for current inhibition
        #M = (I - (1.0 - I)) * (f/2.0)
        factor = kernel[1]
        n_cols = shape[2]
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
    elif init_type == "classic_glorot":
        N = (shape[0] + shape[1]) * 1.0
        bound = 4.0 * np.sqrt(6.0/N)
        params = tf.random.uniform(shape, minval=-bound, maxval=bound, seed=seed)
    elif init_type == "glorot_normal":
        initializer = tf.compat.v1.keras.initializers.glorot_normal()
        params = initializer(shape) #, seed=seed )
    elif init_type == "glorot_uniform":
        initializer = tf.compat.v1.keras.initializers.glorot_uniform()
        params = initializer(shape) #, seed=seed )
    elif init_type == "orthogonal":
        stddev = kernel[1]
        initializer = tf.compat.v1.keras.initializers.orthogonal(gain=stddev)
        params = initializer(shape)
    elif init_type == "truncated_normal" or init_type == "truncated_gaussian" :
        stddev = kernel[1]
        params = tf.random.truncated_normal(shape, stddev=stddev, seed=seed)
    elif init_type == "normal" or init_type == "gaussian":
        stddev = kernel[1]
        params = tf.random.normal(shape, stddev=stddev, seed=seed)
    elif init_type == "uniform":
        scale = kernel[1]
        params = tf.random.uniform(shape, minval=-scale, maxval=scale, seed=seed)
    elif init_type == "alex_uniform": #
        k = 1.0 / (shape[0] * 1.0) # 1/in_features
        bound = np.sqrt(k)
        params = tf.random.uniform(shape, minval=-bound, maxval=bound, seed=seed)
    elif init_type == "unif_scale":
        Phi = np.random.randn(shape[0], shape[1]).astype(np.float32)
        Phi = Phi * np.sqrt(1.0/shape[0])
        params = tf.cast(Phi,dtype=tf.float32)
    else: # zeros
        params = tf.zeros(shape)
    params = tf.cast(params,dtype=tf.float32)
    return params

def create_competiion_matrix(z_dim, lat_type, beta_scale, alpha_scale, n_group, band):
    """
    This function creates a particular matrix to simulate competition via
    self-excitatory and inhibitory synaptic signals.

    Args:
        z_dim: dimensionality of neural group to apply competition to

        lat_type: type of competiton pattern. "lkwta" sets a column/group based
            form of k-WTA style competition and "band" sets a matrix band-based
            form of competition.

        beta_scale: the strength of the cross-unit inhibiton

        alpha_scale: the strength of the self-excitation

        n_group: if lat_type is set to "lkwta", then this ensures that only
            a certain number of neurons are within a competitive group/column

            :Note: z_dim should be divisible by n_group

        band: the band parameter (Note: not fully tested)

    Returns:
        a (z_dim x z_dim) competition matrix
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

def normalize_by_norm(param, proj_mag=1.0, param_axis=0):
    #np.maximum(np.linalg.norm(self.Phi, ord=2, axis=0, keepdims=True), 1e-8)
    p_norm = tf.math.maximum(tf.norm(param, ord=2, axis=param_axis, keepdims=True), 1e-8)
    #print("NGC norm:\n",p_norm)
    param = param * (proj_mag / p_norm)
    return param

def to_one_hot(idx, depth):
    """
    Converts an integer or integer array into a binary one-hot encoding.

    Args:
        idx: an integer or integer list representing the index/indices of the
            chosen category/categories

        depth: total number of actual categories (the dimension K of the encoding)

    Returns:
        a binary one-of-K encoding of the input idx (an N x K vector if len(idx) = N)
    """
    if isinstance(idx, list) == True:
        return tf.cast(tf.one_hot(idx, depth=depth),dtype=tf.float32)
    elif isinstance(idx, np.ndarray) == True:
        idx_ = idx
        if len(idx_.shape) >= 2:
            idx_ = np.squeeze(idx_)
        return tf.cast(tf.one_hot(idx_, depth=depth),dtype=tf.float32)
    enc = tf.cast(tf.one_hot(idx, depth=depth),dtype=tf.float32)
    if len(enc.shape) == 1:
        enc = tf.expand_dims(enc,axis=0)
    return enc

def scale_feat(x, a=-1.0, b=1.0):
    """
    Applies the min-max feature scaling function to input x.

    Args:
        a: the lower bound to scale *x* w/in

        b: the upper bound to scale *x* w/in

    Returns:
        the scaled version of *x*, w/ each value in range [a,b]
    """
    max_x = tf.reduce_max(x,axis=1,keepdims=True)
    min_x = tf.reduce_min(x,axis=1,keepdims=True)
    x_prime = a + ( ( (x - min_x) * (b - a) )/(max_x - min_x) )
    return tf.cast(x_prime, dtype=tf.float32)

def binarize(data, threshold=0.5):
    """
    Converts the vector *data* to its binary equivalent

    Args:
        data: the data to binarize (real-valued)

        threshold: the cut-off point for 0, i.e., if threshold = 0.5, then any
            number/value inside of data < 0.5 is set to 0, otherwise, it is set
            to 1.0

    Returns:
        the binarized equivalent of "data"
    """
    return tf.cast(tf.greater_equal(data, threshold),dtype=tf.float32)


def convert_to_spikes_(x_data, max_spike_rate, dt, sp_div=4.0):
    """
    Converts a vector *x_data* to its approximate Poisson spike equivalent.

    Note: this function is NOT fully tested/integrated yet.

    Args:
        max_spike_rate: firing rate (in Hertz)

        dt: integraton time constant (in milliseconds or ms)

        sp_div: to denominator to convert input data values to a firing frequency

    Returns:
        the binary spike vector form of *x_data*
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
    """
    Applies a filter to data *x_t*.

    Note: this function is NOT fully tested/integrated yet.

    Args:
        x_t:

        x_f:

        dt:

        a:

        filter_type:  (Default = "var_trace")

    Returns:
        the filtered vector form of x_t
    """
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
    """
    Flips the bit values within binary vector *x_b*

    Args:
        x_b: the binary vector to flip

    Returns:
        the flipped binary vector form of x_b
    """
    return (-x_b + 1.0)

def bernoulli(x): # wraps bernoulli sampler as an activation fx
    return sample_bernoulli(x)

def identity(z):
    return z

def d_identity(x):
    return x * 0 + 1.0

def gte(x, val=0.0):
    return tf.cast(tf.greater_equal(x, val),dtype=tf.float32)

def elu(z,alpha=1.0):
    switch = tf.cast(tf.greater_equal(z,0.0),dtype=tf.float32)
    term1 = switch * z
    term2 = (1. - switch) * ((tf.math.exp(z) - 1.0) * alpha)
    # z if z >= 0 else (tf.math.exp(z) - 1.0) * alpha
    return term1 + term2

def d_elu(z,alpha=1.0):
    switch = tf.cast(tf.greater(z,0.0),dtype=tf.float32)
    term1 = switch
    term2 = (1. - switch) * (tf.math.exp(z) * alpha)
    # 1 if z > 0 else tf.math.exp(z) * alpha
    return term1 + term2

def sigmoid(x):
    return tf.nn.sigmoid(x)

def d_sigmoid(x):
    sigm_x = tf.nn.sigmoid(x)
    return (-sigm_x + 1.0) * sigm_x

def inverse_logistic(x, clip_bound=0.03): # 0.03
    """ Inverse logistic link - logit function """
    x_ = x
    if clip_bound > 0.0:
        x_ = tf.clip_by_value(x_, clip_bound, 1.0 - clip_bound)
    return tf.math.log( x_/((1.0 - x_) + 1e-6) )

def sign(x):
    return tf.math.sign(x)

def tanh(x):
    return tf.nn.tanh(x)

def d_tanh(x):
    tanh_x = tf.nn.tanh(x)
    return -(tanh_x * tanh_x) + 1.0

def inverse_tanh(x):
    """ Inverse hyperbolic tangent """
    #m = 0.5 * log ( (ones(size(x)) + x) ./ (ones(size(x)) - x))
    return tf.math.log((1. + x)/(1. - x))

def relu(x):
    return tf.nn.relu(x)

def d_relu(x):
    # df/dx = 1 if x >= 0 else 0
    val = tf.math.greater_equal(x, 0.0)
    return tf.cast(val,dtype=tf.float32) # sign(max(0,x))

def relu6(x):
    return tf.nn.relu6(x)

def d_relu6(x):
    # df/dx = 1 if 0<x<6 else 0
    # I_x = (z >= a_min) *@ (z <= b_max) //create an indicator function  a = 0 b = 6
    Ix1 = tf.cast(tf.math.greater_equal(x, 0.0),dtype=tf.float32)
    Ix2 = tf.cast(tf.math.less_equal(x, 6.0),dtype=tf.float32)
    Ix = Ix1 * Ix2
    return Ix

def softplus(x):
    return tf.nn.softplus(x)

def d_softplus(x):
    return tf.nn.sigmoid(x) # d/dx of softplus = logistic sigmoid

def lrelu(x):
    return tf.nn.leaky_relu(x)

def erf(x):
    return tf.math.erf(x)

def threshold_soft(x, lmda):
    # soft thresholding fx - S(x) = (|x| - lmbda) *@ sign(x)
    return tf.math.maximum(x - lmda, 0.) - tf.math.maximum(-x - lmda, 0.)

def threshold_cauchy(x, lmda):
    # threshold function based on that proposed in: https://arxiv.org/abs/2003.12507
    inner_term = tf.math.sqrt(tf.math.maximum(tf.math.square(x) - lmbda), 0.)
    f = (x + inner_term) * 0.5
    g = (x - inner_term) * 0.5
    term1 = f * tf.greater_equal(x, lmda) # f * (x >= lmda)
    term2 = g * tf.less_equal(x, -lmda) # g * (x <= -lmda)
    return term1 + term2

def softmax(x, tau=0.0):
    """
    Softmax function with overflow control built in directly. Contains optional
    temperature parameter to control sharpness (tau > 1 softens probs, < 1 sharpens --> 0 yields point-mass)

    Args:
        x: a (N x D) input argument (pre-activity) to the softmax operator
        tau: probability sharpening/softening factor

    Returns:
        a (N x D) probability distribution output block
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

def sech(x):
    # 2e^x / (e^2x + 1)
    return (tf.math.exp(x) * 2) / (tf.math.exp(x * 2) + 1)

def sech_sqr(x):
    # sech^2(x) = 1 - tanh^2(x)
    tanh_x = tanh(x)
    return -(tanh_x * tanh_x) + 1.0

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

def calc_modulatory_factor(W):
    """
    Calculate modulatory matrix W_M for W

    Note: this is NOT fully tested/integrated yet
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

def global_contrast_normalization(Xin, s, lmda, epsilon):
    # applies rudimentary global contrast normalization to Xin
    X = Xin
    X_average = np.mean(X)
    X = X - X_average
    # `su` is here the mean, instead of the sum
    contrast = np.sqrt(lmda + np.mean(X**2))
    X = s * X / max(contrast, epsilon)
    return X

def normalize_image(image):
    """
    Maps image array first to [0, image.max() - image.min()]
    then to [0, 1]

    Arg:
        image: the image numpy.ndarray

    Returns:
        image array mapped to [0, 1]
    """
    image = image.astype(float)
    if image.min() != image.max():
        image -= image.min()
    nonzeros = np.nonzero(image)
    image[nonzeros] = image[nonzeros] / image[nonzeros].max()
    return image

def calc_zca_whitening_matrix(X):
    """
    Calculates a ZCA whitening matrix via the Mahalanobis whitening method.

    Note: this is NOT fully tested/integrated yet

    Args:
        X: a design matrix of shape (M x N),
            where rows -> features, columns -> observations

    Returns:
        the resultant (M x M) ZCA matrix
    """
    # Covariance matrix [column-wise variables]: Sigma = (X-mu)' * (X-mu) / N
    sigma = np.cov(X, rowvar=True) # [M x M]
    # Singular Value Decomposition. X = U * np.diag(S) * V
    U,S,V = np.linalg.svd(sigma)
        # U: [M x M] eigenvectors of sigma.
        # S: [M x 1] eigenvalues of sigma.
        # V: [M x M] transpose of U
    # Whitening constant: prevents division by zero
    epsilon = 1e-5
    # ZCA Whitening matrix: U * Lambda * U'
    ZCAMatrix = np.dot(U, np.dot(np.diag(1.0/np.sqrt(S + epsilon)), U.T)) # [M x M]
    return ZCAMatrix

def whiten(X):
    """
    Whitens image X via ZCA whitening

    Note: this is NOT fully tested/integrated yet
    """
    ZCAMatrix = zca_whitening_matrix(X) # get ZCAMatrix
    xZCAMatrix = np.dot(ZCAMatrix, X) # project X onto the ZCAMatrix
    return xZCAMatrix
