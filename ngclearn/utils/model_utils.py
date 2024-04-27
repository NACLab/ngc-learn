import jax
from jax import numpy as jnp, grad, jit, vmap, random, lax, nn
import os, sys
from functools import partial

def pull_equations(controller):
    """
    Extracts the dynamics string of this controller (model/system).

    Args:
        controller: model/system to extract dynamics equation(s) from

    Returns:
        string containing this model/system's dynamics equation(s)
    """
    eqn_set = ""
    for _name in controller.components:
        component = controller.components[_name]
        ## determine if component has an equation and pull it out if so
        for attr in dir(component):
            if not callable(getattr(component, attr)) and not attr.startswith("__"):
                if attr == "equation":
                    eqn = "{}".format(attr) ## extract defined equation
                    eqn_set = "{}\n{}:  {}".format(_name, eqn)
    return eqn_set

@jit
def measure_ACC(mu, y): ## measures/calculates accuracy
    """
    Calculates the accuracy (ACC) given a matrix of predictions and matrix of targets.

    Args:
        mu: prediction (design) matrix

        y: target / ground-truth (design) matrix

    Returns:
        scalar accuracy score
    """
    guess = jnp.argmax(mu, axis=1)
    lab = jnp.argmax(y, axis=1)
    acc = jnp.sum( jnp.equal(guess, lab) )/(y.shape[0] * 1.)
    return acc

def measure_KLD(p_xHat, p_x, preserve_batch=False):
    """
    Measures the (raw) Kullback-Leibler divergence (KLD), assuming that the two
    input arguments contain valid probability distributions (in each row, if
    they are matrices). Note: If batch is preserved, this returns a column
    vector where each row is the KLD(x_pred, x_true) for that row's datapoint.

    | Formula:
    | KLD(p_xHat, p_x) = (1/N) [ sum_i(p_x * jnp.log(p_x)) - sum_i(p_x * jnp.log(p_xHat)) ]
    | where sum_i implies summing across dimensions of vector-space of p_x

    Args:
        p_xHat: predicted probabilities; (N x C matrix, where C is number of categories)

        p_x: ground true probabilities; (N x C matrix, where C is number of categories)

        preserve_batch: if True, will return one score per sample in batch
            (Default: False), otherwise, returns scalar mean score

    Returns:
        an (N x 1) column vector (if preserve_batch=True) OR (1,1) scalar otherwise
    """
    ## numerical control step
    offset = 1e-6
    _p_x = jnp.clip(p_x, offset, 1. - offset)
    _p_xHat = jnp.clip(p_xHat, offset, 1. - offset)
    ## calc raw KLD scores
    N = p_x.shape[1]
    term1 = jnp.sum(_p_x * jnp.log(_p_x), axis=1, keepdims=True) # * (1/N)
    term2 = -jnp.sum(_p_x * jnp.log(_p_xHat), axis=1, keepdims=True) # * (1/N)
    kld = (term1 + term2) * (1/N)
    if preserve_batch == False:
        kld = jnp.mean(kld)
    return kld

@partial(jit, static_argnums=[3])
def measure_CatNLL(p, x, offset=1e-7, preserve_batch=False):
    """
    Measures the negative Categorical log likelihood (Cat.NLL).  Note: If batch is
    preserved, this returns a column vector where each row is the
    Cat.NLL(p, x) for that row's datapoint.

    Args:
        p: predicted probabilities; (N x C matrix, where C is number of categories)

        x: true one-hot encoded targets; (N x C matrix, where C is number of categories)

        offset: factor to control for numerical stability (Default: 1e-7)

        preserve_batch: if True, will return one score per sample in batch
            (Default: False), otherwise, returns scalar mean score

    Returns:
        an (N x 1) column vector (if preserve_batch=True) OR (1,1) scalar otherwise
    """
    p_ = jnp.clip(p, offset, 1.0 - offset)
    loss = -(x * jnp.log(p_))
    nll = jnp.sum(loss, axis=1, keepdims=True) #/(y_true.shape[0] * 1.0)
    if preserve_batch == False:
        nll = jnp.mean(nll)
    return nll #tf.reduce_mean(nll)

@jit
def measure_MSE(mu, x, preserve_batch=False):
    """
    Measures mean squared error (MSE), or the negative Gaussian log likelihood
    with variance of 1.0. Note: If batch is preserved, this returns a column
    vector where each row is the MSE(mu, x) for that row's datapoint.

    Args:
        mu: predicted values (mean); (N x D matrix)

        x: target values (data); (N x D matrix)

        preserve_batch: if True, will return one score per sample in batch
            (Default: False), otherwise, returns scalar mean score

    Returns:
        an (N x 1) column vector (if preserve_batch=True) OR (1,1) scalar otherwise
    """
    diff = mu - x
    se = jnp.square(diff) ## squared error
    mse = jnp.sum(se, axis=1, keepdims=True) # technically se at this point
    if preserve_batch == False:
        mse = jnp.mean(mse) # this is proper mse
    return mse

@jit
def measure_BCE(p, x, offset=1e-7, preserve_batch=False): #1e-10
    """
    Calculates the negative Bernoulli log likelihood or binary cross entropy (BCE).
    Note: If batch is preserved, this returns a column vector where each row is
    the BCE(p, x) for that row's datapoint.

    Args:
        p: predicted probabilities of shape; (N x D matrix)

        x: target binary values (data) of shape; (N x D matrix)

        offset: factor to control for numerical stability (Default: 1e-7)

        preserve_batch: if True, will return one score per sample in batch
            (Default: False), otherwise, returns scalar mean score

    Returns:
        an (N x 1) column vector (if preserve_batch=True) OR (1,1) scalar otherwise
    """
    p_ = jnp.clip(p, offset, 1 - offset)
    bce = -jnp.sum(x * jnp.log(p_) + (1.0 - x) * jnp.log(1.0 - p_),axis=1, keepdims=True)
    if preserve_batch == False:
        bce = jnp.mean(bce)
    return bce

def create_function(fun_name, args=None):
    """
    Activation function creation routine.

    Args:
        fun_name: string name of activation function to produce
            (Currently supports: "tanh", "relu", "lrelu", "identity")

    Returns:
        function fx, first derivative of function (w.r.t. input) dfx
    """
    fx = None
    dfx = None
    if fun_name == "tanh":
        fx = tanh
        dfx = d_tanh
    elif fun_name == "sigmoid":
        fx = sigmoid
        dfx = d_sigmoid
    elif fun_name == "relu":
        fx = relu
        dfx = d_relu
    elif fun_name == "lrelu":
        fx = lrelu
        dfx = d_lrelu
    elif fun_name == "relu6":
        fx = relu6
        dfx = d_relu6
    elif fun_name == "softplus":
        fx = softplus
        dfx = d_softplus
    elif fun_name == "unit_threshold":
        fx = threshold ## default threshold is 1 (thus unit)
        dfx = d_threshold ## STE approximation
    elif "heaviside" in fun_name:
        fx = heaviside
        dfx = d_heaviside ## STE approximation
    elif fun_name == "identity":
        fx = identity
        dfx = d_identity
    else:
        raise RuntimeError(
            "Activition function (" + fun_name + ") is not recognized/supported!"
            )
    return fx, dfx


def initialize_params(dkey, initKernel, shape):
    """
    Creates the intiial condition values for a parameter tensor.

    Args:
        dkey: PRNG key to control determinism of this routine

        initKernel: triplet/3-tuple with 1st element as a string calling the name
            of initialization scheme to use

            :Note: Currently supported kernel schemes include:
                ("hollow", off_diagonal_scale, ~ignored~);
                ("eye", diagonal_scale, ~ignored~);
                ("uniform", min_val, max_val);
                ("gaussian", mu, sigma) OR ("normal", mu, sigma);
                ("constant", magnitude, ~ignored~)

        shape: tuple containing the dimensions/shape of the tensor to initialize

    Returns:
        output (tensor) value
    """
    initType, *args = initKernel # get out arguments of initialization kernel
    params = None
    if initType == "hollow":
        eyeScale, _ = args
        params = (1. - jnp.eye(N=shape[0], M=shape[1])) * eyeScale
    elif initType == "eye":
        eyeScale, _ = args
        params = jnp.eye(N=shape[0], M=shape[1]) * eyeScale
    elif initType == "uniform": ## uniformly distributed values
        lb, ub = args
        params = random.uniform(dkey, shape, minval=lb, maxval=ub)
    elif initType == "gaussian" or initType == "normal": ## gaussian distributed values
        mu, sigma = args
        params = random.normal(dkey, shape) * sigma + mu
    elif initType == "constant": ## constant value(s)
        scale, _ = args
        params = jnp.ones(shape) * scale
    else:
        raise RuntimeError(
            "Initialization scheme (" + initType + ") is not recognized/supported!"
            )
    return params

@partial(jit, static_argnums=[2, 3])
def normalize_matrix(M, wnorm, order=1, axis=0):
    """
    Normalizes the values in matrix to have a particular norm across each vector span.

    Args:
        M: (2D) matrix to normalize

        wnorm: target norm for each

        order: order of norm to use in normalization (Default: 1);
            note that `ord=1` results in the L1-norm, `ord=2` results in the L2-norm

        axis: 0 (apply to column vectors), 1 (apply to row vectors)

    Returns:
        a normalized value matrix
    """
    if order == 2: ## denominator is L2 norm
        wOrdSum = jnp.square(jnp.sum(jnp.square(M), axis=axis, keepdims=True))
    else: ## denominator is L1 norm
        wOrdSum = jnp.sum(jnp.abs(M), axis=axis, keepdims=True)
    m = (wOrdSum == 0.).astype(dtype=jnp.float32)
    wOrdSum = wOrdSum * (1. - m) + m
    #wAbsSum[wAbsSum == 0.] = 1.
    _M = M * (wnorm/wOrdSum)
    return _M

@jit
def clamp_min(x, min_val):
    """
    Clamps values in data x that exceed a minimum value to that value.

    Args:
        x: data to lower-bound clamp

        min_val: minimum value threshold

    Returns:
        x with minimum clamped values
    """
    mask = (x > min_val).astype(jnp.float32)
    _x = x * mask + (1. - mask) * min_val
    return _x

@jit
def clamp_max(x, max_val):
    """
    Clamps values in data x that exceed a maximum value to that value.

    Args:
        x: data to upper-bound clamp

        max_val: maximum value threshold

    Returns:
        x with maximum clamped values
    """
    # condition = torch.bitwise_or(torch.le(a, max), a_isnan)  # type: ignore[arg-type]
    #a = torch.where(condition, a, max)
    mask = (x < max_val).astype(jnp.float32)
    _x = x * mask + (1. - mask) * max_val
    return _x


@jit
def one_hot(P):
    """
    Converts a matrix of probabilities to a corresponding binary one-hot matrix
    (each row is a one-hot encoding).

    Args:
        P: a probability matrix where each row corresponds to a particular
            data probability vector

    Returns:
        the one-hot encoding (matrix) of probabilities in P
    """
    nC = P.shape[1] # compute number of dimensions/classes
    p_t = jnp.argmax(P, axis=1)
    return nn.one_hot(p_t, num_classes=nC, dtype=jnp.float32)

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
    return (data > threshold).astype(jnp.float32)

@jit
def identity(x):
    """
    The identity function: x = f(x).

    Args:
        x: input (tensor) value

    Returns:
        output (tensor) value
    """
    return x + 0

@jit
def d_identity(x):
    """
    Derivative of the identity function.

    Args:
        x: input (tensor) value

    Returns:
        output (tensor) derivative value (with respect to input argument)
    """
    return x * 0 + 1.

@jit
def relu(x):
    """
    The linear rectifier: max(0, x) = f(x).

    Args:
        x: input (tensor) value

    Returns:
        output (tensor) value
    """
    return nn.relu(x)

@jit
def d_relu(x):
    """
    Derivative of the linear rectifier.

    Args:
        x: input (tensor) value

    Returns:
        output (tensor) derivative value (with respect to input argument)
    """
    return (x >= 0.).astype(jnp.float32)

@jit
def tanh(x):
    """
    The hyperbolic tangent function.

    Args:
        x: input (tensor) value

    Returns:
        output (tensor) value
    """
    return nn.tanh(x)

@jit
def d_tanh(x):
    """
    Derivative of the hyperbolic tangent function.

    Args:
        x: input (tensor) value

    Returns:
        output (tensor) derivative value (with respect to input argument)
    """
    tanh_x = nn.tanh(x)
    return -(tanh_x * tanh_x) + 1.0

@jit
def inverse_tanh(x):
    """
    The inverse hyperbolic tangent.

    Args:
        x: data to transform via inverse hyperbolic tangent

        clip_bound: pre-processing lower/upper bounds to enforce on data
            before applying inverse hyperbolic tangent

    Returns:
        x transformed via inverse hyperbolic tangent
    """
    #m = 0.5 * log ( (ones(size(x)) + x) ./ (ones(size(x)) - x))
    return jnp.log((1. + x)/(1. - x))

@jit
def lrelu(x): ## activation fx
    """
    The leaky linear rectifier: max(0, x) if x >= 0, 0.01 * x if x < 0 = f(x).

    Args:
        x: input (tensor) value

    Returns:
        output (tensor) value
    """
    return nn.leaky_relu(x)

@jit
def d_lrelu(x): ## deriv of fx (dampening function)
    """
    Derivative of the leaky linear rectifier.

    Args:
        x: input (tensor) value

    Returns:
        output (tensor) derivative value (with respect to input argument)
    """
    m = (x >= 0.).astype(jnp.float32)
    dx = m + (1. - m) * 0.01
    return dx

@jit
def relu6(x):
    """
    The linear rectifier upper bounded at the value of 6: min(max(0, x), 6.).

    Args:
        x: input (tensor) value

    Returns:
        output (tensor) value
    """
    return nn.relu6(x)

@jit
def d_relu6(x):
    """
    Derivative of the bounded leaky linear rectifier (upper bounded at 6).

    Args:
        x: input (tensor) value

    Returns:
        output (tensor) derivative value (with respect to input argument)
    """
    # df/dx = 1 if 0<x<6 else 0
    # I_x = (z >= a_min) *@ (z <= b_max) //create an indicator function  a = 0 b = 6
    Ix1 = (x > 0.).astype(jnp.float32) #tf.cast(tf.math.greater_equal(x, 0.0),dtype=tf.float32)
    Ix2 = (x <= 6.).astype(jnp.float32) #tf.cast(tf.math.less_equal(x, 6.0),dtype=tf.float32)
    Ix = Ix1 * Ix2
    return Ix

@jit
def softplus(x):
    """
    The softplus elementwise function.

    Args:
        x: input (tensor) value

    Returns:
        output (tensor) value
    """
    return nn.softplus(x)

@jit
def d_softplus(x):
    """
    Derivative of the softplus function.

    Args:
        x: input (tensor) value

    Returns:
        output (tensor) derivative value (with respect to input argument)
    """
    ## d/dx of softplus = logistic sigmoid
    return nn.sigmoid(x)

@jit
def threshold(x, thr=1.):
    return (x >= thr).astype(jnp.float32)

@jit
def d_threshold(x, thr=1.):
    return x * 0. + 1. ## straight-thru estimator

@jit
def heaviside(x):
    return (x >= 0.).astype(jnp.float32)

@jit
def d_heaviside(x):
    return x * 0. + 1. ## straight-thru estimator

@jit
def sigmoid(x):
    return nn.sigmoid(x)

@jit
def d_sigmoid(x):
    sigm_x = nn.sigmoid(x) ## pre-compute once
    return sigm_x * (1. - sigm_x)

@jit
def inverse_logistic(x, clip_bound=0.03): # 0.03
    """
    The inverse logistic link - logit function.

    Args:
        x: data to transform via inverse logistic function

        clip_bound: pre-processing lower/upper bounds to enforce on data
            before applying inverse logistic

    Returns:
        x transformed via inverse logistic function
    """
    x_ = x
    if clip_bound > 0.0:
        x_ = jnp.clip(x_, clip_bound, 1.0 - clip_bound)
    return jnp.log( x_/((1.0 - x_) + 1e-6) )

@jit
def softmax(x, tau=0.0):
    """
    Softmax function with overflow control built in directly. Contains optional
    temperature parameter to control sharpness
    (tau > 1 softens probs, < 1 sharpens --> 0 yields point-mass).

    Args:
        x: a (N x D) input argument (pre-activity) to the softmax operator

        tau: probability sharpening/softening factor

    Returns:
        a (N x D) probability distribution output block
    """
    if tau > 0.0:
        x = x / tau
    max_x = jnp.max(x, axis=1, keepdims=True)
    exp_x = jnp.exp(x - max_x)
    return exp_x / jnp.sum(exp_x, axis=1, keepdims=True)

def threshold_soft(x, lmbda):
    """
    A soft threshold routine applied to each dimension of input

    Args:
        x: data to apply threshold function over

        lmbda: scalar to control strength/influence of thresholding

    Returns:
        thresholded x
    """
    # soft thresholding fx - S(x) = (|x| - lmbda) *@ sign(x)
    ## legacy ngclearn: tf.math.maximum(x - lmbda, 0.) - tf.math.maximum(-x - lmbda, 0.)
    return jnp.maximum(x - lmbda, 0.) - jnp.maximum(-x - lmbda, 0.)

def threshold_cauchy(x, lmbda):
    """
    A Cauchy distributional threshold routine applied to each dimension of input

    Args:
        x: data to apply threshold function over

        lmbda: scalar to control strength/influence of Cauchy thresholding

    Returns:
        thresholded x
    """
    # threshold function based on that proposed in: https://arxiv.org/abs/2003.12507
    inner_term = jnp.sqrt(jnp.maximum(jnp.square(x) - lmbda), 0.)
    f = (x + inner_term) * 0.5
    g = (x - inner_term) * 0.5
    term1 = f * (x >= lmbda).astype(jnp.float32) ## f * (x >= lmda)
    term2 = g * (x <= -lmbda).astype(jnp.float32) ## g * (x <= -lmda)
    return term1 + term2

@jit
def drop_out(dkey, input, rate=0.0):
    """
    Applies a drop-out transform to an input matrix.

    Args:
        dkey: Jax randomness key for this operator

        input: data to apply random/drop-out mask to

        rate: probability of a dimension being dropped

    Returns:
        output as well as binary mask
    """
    eps = random.uniform(dkey, (input.shape[0],input.shape[1]),
                         minval=0.0, maxval=1.0)
    mask = (eps <= (1.0 - rate)).astype(jnp.float32)
    mask = mask * (1.0 / (1.0 - rate)) ## apply inverted dropout scheme
    output = input * mask
    return output, mask
