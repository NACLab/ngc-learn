import jax
import numpy as np
from ngclearn.utils.analysis.probe import Probe
from ngclearn.utils.model_utils import drop_out, softmax, gelu, layer_normalize
from ngclearn.utils.optim import adam
from jax import jit, random, numpy as jnp, lax, nn
from functools import partial as bind

def masked_fill(x: jax.Array, mask: jax.Array, value=0) -> jax.Array:
    """
    Return an output with masked condition, with non-masked value
    be the other value

    Args:
        x (jax.Array): _description_
        mask (jax.Array): _description_
        value (int, optional): _description_. Defaults to 0.

    Returns:
        jax.Array: _description_
    """
    return jnp.where(mask, jnp.broadcast_to(value, x.shape), x)

@bind(jax.jit, static_argnums=[4, 5])
def cross_attention(params: tuple, x1: jax.Array, x2: jax.Array, mask: jax.Array, n_heads: int=8, dropout_rate: float=0.0):
    B, T, Dq = x1.shape # The original shape
    _, S, Dkv = x2.shape
    # in here we attend x2 to x1
    Wq, bq, Wk, bk, Wv, bv, Wout, bout = params
    # projection
    q = x1 @ Wq + bq # normal linear transformation (B, T, D)
    k = x2 @ Wk + bk # normal linear transformation (B, S, D)
    v = x2 @ Wv + bv # normal linear transformation (B, S, D)
    hidden = q.shape[-1]
    _hidden = hidden // n_heads
    q = q.reshape((B, T, n_heads, _hidden)).transpose([0, 2, 1, 3]) # (B, H, T, D)
    k = k.reshape((B, S, n_heads, _hidden)).transpose([0, 2, 1, 3]) # (B, H, T, D)
    v = v.reshape((B, S, n_heads, _hidden)).transpose([0, 2, 1, 3]) # (B, H, T, D)
    score = jnp.einsum("BHTE,BHSE->BHTS", q, k) / jnp.sqrt(_hidden) # Q @ KT / ||d||; d = D // n_heads
    if mask is not None:
        Tq, Tk = q.shape[2], k.shape[2]
        assert mask.shape == (B, Tq, Tk), (mask.shape, (B, Tq, Tk))
        _mask = mask.reshape((B, 1, Tq, Tk)) # 'b tq tk -> b 1 tq tk'
        score = masked_fill(score, _mask, value=-jnp.inf) # basically masking out all must-unattended values
    score = jax.nn.softmax(score, axis=-1) # (B, H, T, S)
    score = score.astype(q.dtype) # (B, H, T, S)
    if dropout_rate > 0.:
        score = drop_out(input=score, rate=dropout_rate) ## NOTE: normally you apply dropout here
    attention = jnp.einsum("BHTS,BHSE->BHTE", score, v) # (B, T, H, E)
    attention = attention.transpose([0, 2, 1, 3]).reshape((B, T, -1)) # (B, T, H, E) => (B, T, D)
    return attention @ Wout + bout # (B, T, Dq)

@bind(jax.jit, static_argnums=[3, 4, 5, 6])
def run_attention_probe(params, encodings, mask, n_heads: int, dropout: float = 0.0, use_LN=False, use_softmax=True):
    # encoded_image_feature: (B, hw, dim)
    #learnable_query, *_params) = params
    learnable_query, Wq, bq, Wk, bk, Wv, bv, Wout, bout, Whid, bhid, Wln_mu, Wln_scale, Wy, by = params
    attn_params = (Wq, bq, Wk, bk, Wv, bv, Wout, bout)
    features = cross_attention(attn_params, learnable_query, encodings, mask, n_heads, dropout)
    features = features[:, 0]  # (B, 1, dim) => (B, dim)
    hids = jnp.matmul((features + learnable_query[:, 0]), Whid) + bhid
    hids = gelu(hids)
    if use_LN: ## normalize hidden layer output of probe predictor
        hids = layer_normalize(hids, Wln_mu, Wln_scale)
    outs = jnp.matmul(hids, Wy) + by
    if use_softmax: ## apply softmax output nonlinearity
        outs = softmax(outs)
    return outs, features

@bind(jax.jit, static_argnums=[4, 5, 6, 7])
def eval_attention_probe(params, encodings, labels, mask, n_heads: int, dropout: float = 0.0, use_LN=False, use_softmax=True):
    # encodings: (B, hw, dim)
    outs, _ = run_attention_probe(params, encodings, mask, n_heads, dropout, use_LN, use_softmax)
    if use_softmax: ## Multinoulli log likelihood for 1-of-K predictions
        L = -jnp.mean(jnp.sum(jnp.log(outs) * labels, axis=1, keepdims=True))
    else: ## MSE for real-valued outputs
        L = jnp.mean(jnp.sum(jnp.square(outs - labels), axis=1, keepdims=True))
    return L, outs #, features

class AttentiveProbe(Probe):
    """
    Args:
        dkey: init seed key

        source_seq_length: length of input sequence (e.g., height x width of the image feature)

        input_dim: input dimensionality of probe

        out_dim: output dimensionality of probe

        num_heads: number of cross-attention heads

        head_dim: output dimensionality of each cross-attention head

        target_seq_length: to pool, we set it at one (or map the source sequence to the target sequence of length 1)

        learnable_query_dim: target sequence dim (output dimension of cross-attention portion of probe)

        batch_size: size of batches to process per internal call to update (or process)

        hid_dim: dimensionality of hidden layer(s) of MLP portion of probe

        use_LN: should layer normalization be used within MLP portions of probe or not?

        use_softmax: should a softmax be applied to output of probe or not?

    """
    def __init__(
            self, dkey, source_seq_length, input_dim, out_dim, num_heads=8, head_dim=64,
            target_seq_length=1, learnable_query_dim=31, batch_size=1, hid_dim=32, use_LN=True, use_softmax=True, **kwargs
    ):
        super().__init__(dkey, batch_size, **kwargs)
        self.dkey, *subkeys = random.split(self.dkey, 12)
        self.num_heads = num_heads
        self.source_seq_length = source_seq_length
        self.input_dim = input_dim
        self.out_dim = out_dim
        self.use_softmax = use_softmax
        self.use_LN = use_LN

        sigma = 0.05
        ## cross-attention parameters
        Wq = random.normal(subkeys[0], (learnable_query_dim, head_dim)) * sigma
        bq = random.normal(subkeys[1], (1, head_dim)) * sigma
        Wk = random.normal(subkeys[2], (input_dim, head_dim)) * sigma
        bk = random.normal(subkeys[3], (1, head_dim)) * sigma
        Wv = random.normal(subkeys[4], (input_dim, head_dim)) * sigma
        bv = random.normal(subkeys[5], (1, head_dim)) * sigma
        Wout = random.normal(subkeys[6], (head_dim, learnable_query_dim)) * sigma
        bout = random.normal(subkeys[7], (1, learnable_query_dim)) * sigma
        #params = (Wq, bq, Wk, bk, Wv, bv, Wout, bout)
        learnable_query = jnp.zeros((batch_size, 1, learnable_query_dim))  # (B, T, D)
        #self.all_params = (learnable_query, *params)
        self.mask = np.zeros((batch_size, target_seq_length, source_seq_length)).astype(bool) ## mask tensor
        ## MLP parameters
        Whid = random.normal(subkeys[8], (learnable_query_dim, hid_dim)) * sigma
        bhid = random.normal(subkeys[9], (1, hid_dim)) * sigma
        Wln_mu = jnp.zeros((1, hid_dim))
        Wln_scale = jnp.ones((1, hid_dim))
        Wy = random.normal(subkeys[8], (hid_dim, out_dim)) * sigma
        by = random.normal(subkeys[9], (1, out_dim)) * sigma
        #mlp_params = (Whid, bhid, Wln_mu, Wln_scale, Wy, by)
        self.probe_params = (learnable_query, Wq, bq, Wk, bk, Wv, bv, Wout, bout, Whid, bhid, Wln_mu, Wln_scale, Wy, by)

        ## set up gradient calculator
        self.grad_fx = jax.value_and_grad(eval_attention_probe, argnums=0, has_aux=True)
        ## set up update rule/optimizer
        self.optim_params = adam.adam_init(self.probe_params)
        self.eta = 0.001

    def process(self, embedding_sequence):
        outs, feats = run_attention_probe(
            self.probe_params, embedding_sequence, self.mask, self.num_heads, 0.0, use_LN=self.use_LN,
            use_softmax=self.use_softmax
        )
        return outs

    def update(self, embedding_sequence, labels):
        ## compute partial derivatives / adjustments to probe parameters
        outputs, grads = self.grad_fx(
            self.probe_params, embedding_sequence, labels, self.mask, self.num_heads, dropout=0., use_LN=self.use_LN,
            use_softmax=self.use_softmax
        )
        loss, predictions = outputs
        ## adjust parameters of probe
        self.optim_params, self.probe_params = adam.adam_step(
            self.optim_params, self.probe_params, grads, eta=self.eta
        )
        return loss, predictions

