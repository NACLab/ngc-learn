import jax
from jax import numpy as jnp, grad, jit, vmap, random, lax, nn
import os, sys
from functools import partial

def get_integrator_code(integrationType):
    """
    Convenience function for mapping integrator type string to ngc-learn's
    internal integer code value.

    Args:
        integrationType: string indicating integrator type
            (`euler` or `rk1`, `rk2`)

    Returns:
        integator type integer code
    """
    intgFlag = 0 ## Default is Euler (RK1)
    if integrationType == "midpoint" or integrationType == "rk2":
        intgFlag = 1
    elif integrationType == "rk3": ## Runge-Kutte 3rd order code
        intgFlag = 2
    elif integrationType == "rk4": ## Runge-Kutte 4rd order code
        intgFlag = 3
    return intgFlag

def pull_equation(component):
    """
    Extracts the dynamics string of this component.

    Args:
        component: component to extract dynamics equation(s) from

    Returns:
        string containing this component's dynamics equation(s)
    """
    eqn = ""
    for attr in dir(component):
        if not callable(getattr(component, attr)) and not attr.startswith("__"):
            if attr == "equation":
                eqn = "{}".format(attr)
    return eqn

@jit
def calc_acc(mu, y): ## calculates accuracy
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

def measure_CatNLL(p, x, epsilon=0.0000001): #1e-7):
    """
    Measures the negative Categorical log likelihood

    Args:
        p: predicted probabilities; (N x C matrix, where C is number of categories)

        x: true one-hot encoded targets; (N x C matrix, where C is number of categories)

    Returns:
        an (N x 1) column vector, where each row is the Cat.NLL(x_pred, x_true)
        for that row's datapoint
    """
    p_ = jnp.clip(p, epsilon, 1.0 - epsilon)
    loss = -(x * jnp.log(p_))
    nll = jnp.sum(loss, axis=1, keepdims=True) #/(y_true.shape[0] * 1.0)
    return nll #tf.reduce_mean(nll)

def measure_MSE(mu, x):
    """
    Measures mean squared error (MSE), or the negative Gaussian log likelihood
    with variance of 1.0.

    Args:
        mu: predicted values (mean); (N x D matrix)

        x: target values (data); (N x D matrix)

    Returns:
        an (N x 1) column vector, where each row is the MSE(x_pred, x_true) for that row's datapoint
    """
    diff = mu - x
    se = jnp.square(diff) #diff * diff # squared error
    # NLL = -( -se )
    return jnp.sum(se, axis=1, keepdims=True) # tf.math.reduce_mean(se)

def measure_BCE(p, x, offset=1e-7): #1e-10
    """
    Calculates the negative Bernoulli log likelihood or binary cross entropy (BCE).

    Args:
        p: predicted probabilities of shape; (N x D matrix)

        x: target binary values (data) of shape; (N x D matrix)

    Returns:
        an (N x 1) column vector, where each row is the BCE(p, x) for that row's datapoint
    """
    p_ = jnp.clip(p, offset, 1 - offset)
    return -jnp.sum(x * jnp.log(p_) + (1.0 - x) * jnp.log(1.0 - p_),axis=1, keepdims=True)

def create_function(fun_name):
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
    elif fun_name == "relu":
        fx = relu
        dfx = d_relu
    elif fun_name == "lrelu":
        fx = lrelu
        dfx = d_lrelu
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

        initKernel: tuple with 1st element as a string calling the name of
            initialization to use (Currently supported: "hollow", "eye", "uniform")

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
    elif initType == "constant": ## constant value(s)
        scale, _ = args
        params = jnp.ones(shape) * scale
    else:
        raise RuntimeError(
            "Initialization scheme (" + initType + ") is not recognized/supported!"
            )
    return params

@partial(jit, static_argnums=[2, 3])
def normalize_matrix(M, wnorm, ord=1, axis=0):
    """
    Normalizes the values in matrix to have a particular norm across each vector span.

    Args:
        M: (2D) matrix to normalize

        wnorm: target norm for each

        ord: order of norm to use in normalization (Default: 1);
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
