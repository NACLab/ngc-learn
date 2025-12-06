import jax
import numpy as np
from ngclearn.utils.analysis.probe import Probe
from ngclearn.utils.model_utils import drop_out, softmax, layer_normalize
from jax import jit, random, numpy as jnp, lax, nn
from functools import partial as bind
from ngclearn.utils.distribution_generator import DistributionGenerator
from ngclearn.utils.optim import adam, sgd

@bind(jax.jit, static_argnums=[2, 3])
def run_linear_probe(params, x, use_softmax=False, use_LN=False):
    Wln_mu, Wln_scale, W, b = params
    _x = x
    if use_LN:  ## normalize input vector to probe predictor
        _x = layer_normalize(_x, Wln_mu, Wln_scale)
    y_mu = (jnp.matmul(_x, W) + b)
    if use_softmax:
        y_mu = softmax(y_mu)
    return y_mu

@bind(jax.jit, static_argnums=[3, 4])
def eval_linear_probe(params, x, y, use_softmax=True, use_LN=False):
    y_mu = run_linear_probe(params, x, use_softmax=use_softmax, use_LN=use_LN)
    e = y_mu - y
    if use_softmax: ## Multinoulli log likelihood for 1-of-K predictions
        L = -jnp.mean(jnp.sum(jnp.log(y_mu) * y, axis=1, keepdims=True))
    else: ## MSE for real-valued outputs
        L = jnp.sum(jnp.square(e)) * 1./x.shape[0]
    return L, y_mu
    #return y_mu, L, e

# @bind(jax.jit, static_argnums=[6, 7])
# def calc_linear_probe_grad(x, y, params, eta, decay=0., l1_decay=0., use_softmax=False, use_LN=False):
#     y_mu, L, e = eval_linear_probe(params, x, y, use_softmax=use_softmax, use_LN=use_LN)
#     Wln_mu, Wln_scale, W, b = params
#     dW = jnp.matmul(x.T, e) + W * decay/eta + jnp.abs(W) * 0.5 *  l1_decay/eta
#     db = jnp.sum(e, axis=0, keepdims=True)
#     dW = dW * (1. / x.shape[0])
#     db = db * (1. / x.shape[0])
#     return y_mu, L, [dW, db]

# @jit
# def update_linear_probe(x, y, params, eta, decay=0., l1_decay=0., use_softmax=False):
#     y_mu, L, e = run_linear_probe(x, params, use_softmax=use_softmax)
#     W, b = params
#     dW = jnp.matmul(x.T, e)
#     db = jnp.sum(e, axis=0, keepdims=True)
#     W = W - dW * eta/x.shape[0] - W * decay/x.shape[0] - jnp.abs(W) * 0.5 *  l1_decay/x.shape[0]
#     b = b - db * eta/x.shape[0]
#     return y_mu, L, [W, b]

class LinearProbe(Probe):
    """
    This implements a regularized linear probe, which is useful for evaluating the quality of 
    encodings/embeddings in light of some superivsory downstream data (e.g., label one-hot 
    encodings or real-valued vector regression targets). 
    Note that this probe allows for configurable Elastic-net (L1+L2) regularization.

    Args:
        dkey: init seed key

        source_seq_length: length of input sequence (e.g., height x width of the image feature)

        input_dim: input dimensionality of probe

        out_dim: output dimensionality of probe

        batch_size: size of batches to process per internal call to update (or process)

        use_LN: should layer normalization be used on incoming input vectors given to this probe?

        use_softmax: should a softmax be applied to output of probe or not?

    """
    def __init__(
            self, dkey, source_seq_length, input_dim, out_dim, batch_size=1, use_LN=False, use_softmax=False, **kwargs
    ):
        super().__init__(dkey, batch_size, **kwargs)
        self.dkey, *subkeys = random.split(self.dkey, 3)
        self.source_seq_length = source_seq_length
        self.input_dim = input_dim
        self.out_dim = out_dim
        self.use_softmax = use_softmax
        self.use_LN = use_LN
        self.l2_decay = 0.0001
        self.l1_decay = 0.000025
        # eta = 0.05 for SGD, batch_size=2000

        ## set up classifier
        flat_input_dim = input_dim * source_seq_length
        weight_init = DistributionGenerator.fan_in_gaussian() #dist.fan_in_gaussian()  # dist.gaussian(mu=0., sigma=0.05)  # 0.02)
        Wln_mu = jnp.zeros((1, flat_input_dim))
        Wln_scale = jnp.ones((1, flat_input_dim))
        W = weight_init((flat_input_dim, out_dim), subkeys[0]) #dist.initialize_params(subkeys[0], weight_init, (flat_input_dim, out_dim))
        b = jnp.zeros((1, out_dim))
        self.probe_params = [Wln_mu, Wln_scale, W, b]

        ## set up update rule/optimizer
        ## set up gradient calculator
        self.grad_fx = jax.value_and_grad(eval_linear_probe, argnums=0, has_aux=True)
        self.optim_params = adam.adam_init(self.probe_params)
        self.eta = 0.001

    def process(self, embeddings, dkey=None):
        _embeddings = embeddings
        if len(_embeddings.shape) > 2: ## we flatten a sequence batch to 2D for a linear probe
            flat_dim = embeddings.shape[1] * embeddings.shape[2]
            _embeddings = jnp.reshape(_embeddings, (embeddings.shape[0], flat_dim))
        outs = run_linear_probe(self.probe_params, _embeddings, use_softmax=self.use_softmax, use_LN=self.use_LN)
        return outs

    def update(self, embeddings, labels, dkey=None):
        _embeddings = embeddings
        if len(_embeddings.shape) > 2:
            flat_dim = embeddings.shape[1] * embeddings.shape[2]
            _embeddings = jnp.reshape(_embeddings, (embeddings.shape[0], flat_dim))
        ## compute adjustments to probe parameters
        # predictions, loss, grads = calc_linear_probe_grad(
        #     self.probe_params, _embeddings, labels, self.eta, decay=self.l2_decay, l1_decay=self.l1_decay,
        #     use_softmax=self.use_softmax, use_LN=self.use_LN
        # )
        outputs, grads = self.grad_fx(
            self.probe_params, _embeddings, labels, use_softmax=self.use_softmax, use_LN=self.use_LN
        )
        loss, predictions = outputs
        ## adjust parameters of probe
        self.optim_params, self.probe_params = adam.adam_step(
            self.optim_params, self.probe_params, grads, eta=self.eta
        )
        return loss, predictions

