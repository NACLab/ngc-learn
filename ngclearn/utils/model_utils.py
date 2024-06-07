"""
General modeling utility routines and co-routines. This contains useful
commonly jit-i-fied mathematical functions and operations needed to design
and develop ngc-learn internal components.
"""
import jax
from jax import numpy as jnp, grad, jit, vmap, random, lax, nn
from jax.lax import scan as _scan
from ngcsimlib.utils import Get_Compartment_Batch, Set_Compartment_Batch, get_current_context
import os, sys
from functools import partial
import numpy as np

def tensorstats(tensor):
    """
    Prints tensor statistics (debugging tool).

    Args:
        tensor: argument tensor object to examine

    Returns:
        useful statistics to print to I/O
    """
    if isinstance(tensor, (np.ndarray, jax.Array, jnp.ndarray)):
        _tensor = np.asarray(tensor)
        return {
            'mean': _tensor.mean(),
            'std': _tensor.std(),
            'mag': np.abs(_tensor).max(),
            'min': _tensor.min(),
            'max': _tensor.max(),
        }
    elif isinstance(tensor, (list, tuple, dict)):
        try:
            values, _ = jax.tree.flatten(jax.tree.map(lambda x: x.flatten(), tensor))
            values = np.asarray(np.stack(values))
            return {
                'mean': values.mean(),
                'std': values.std(),
                'mag': np.abs(values).max(),
                'min': values.min(),
                'max': values.max(),
            }
        except:
            return None
    else:
        return None

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

@partial(jit, static_argnums=[2, 3, 4])
def normalize_matrix(M, wnorm, order=1, axis=0, scale=1.):
    """
    Normalizes the values in matrix to have a particular norm across each vector span.

    Args:
        M: (2D) matrix to normalize

        wnorm: target norm for each

        order: order of norm to use in normalization (Default: 1);
            note that `ord=1` results in the L1-norm, `ord=2` results in the L2-norm

        axis: 0 (apply to column vectors), 1 (apply to row vectors)

        scale: step modifier to produce the projected matrix

    Returns:
        a normalized value matrix
    """
    if order == 2: ## denominator is L2 norm
        wOrdSum = jnp.maximum(jnp.sum(jnp.square(M), axis=axis, keepdims=True), 1e-8)
    else: ## denominator is L1 norm
        wOrdSum = jnp.maximum(jnp.sum(jnp.abs(M), axis=axis, keepdims=True), 1e-8)
    m = (wOrdSum == 0.).astype(dtype=jnp.float32)
    wOrdSum = wOrdSum * (1. - m) + m #wAbsSum[wAbsSum == 0.] = 1.
    # _M = M * (wnorm/wOrdSum)
    dM = ((wnorm/wOrdSum) - 1.) * M
    _M = M + dM * scale
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

@jit
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

@jit
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


def scanner(fn):
    """
    A wrapper for Jax's scanner that handles the "getting" of the current
    state and "setting" of the final state to and from the model.

    | @scanner
    | def process(current_state, args):
    |    t = args[0]
    |    dt = args[1]
    |    current_state = model.advance_state(current_state, t, dt)
    |    current_state = model.evolve(current_state, t, dt)
    |    return current_state, (current_state[COMPONENT.COMPARTMENT.path], ...)
    |
    | outputs = models.process(jnp.array([[ARG0, ARG1] for i in range(NUM_LOOPS)]))

    | Notes on the scanner function call:
    | 1) `current_state` is a hash-map mapped to all compartment values by path
    | 2) `args` is the external arguments defined in the passed Jax array
    | 3) `outputs` is a tuple containing time-concatenated Jax arrays of the
    |     compartment statistics you want tracked

    Args:
        fn: function that is executed at every time step of a Jax-unrolled loop,
            it must take in the current state and external arguments

    Returns:
        wrapped (fast) function that is Jax-scanned/jit-i-fied
    """
    def _scanned(_xs):
        vals, stacked = _scan(fn, init=Get_Compartment_Batch(), xs=_xs)
        Set_Compartment_Batch(vals)
        return stacked

    if get_current_context() is not None:
        get_current_context().__setattr__(fn.__name__, _scanned)
    return _scanned
