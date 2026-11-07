# %%
from jax import random, numpy as jnp, nn
from ngclearn.components.jaxComponent import JaxComponent
from ngclearn.utils.distribution_generator import DistributionGenerator
from ngcsimlib.logger import info
from ngclearn import compilable, Compartment

def _create_attention_synapses(key, shape, n_patch, use_value=True, init_type=DistributionGenerator.fan_in_gaussian()):
    key, *subkey = random.split(key, 10)

    patch_shape = (shape[0] // n_patch, shape[1] // n_patch)
    di, dj = patch_shape
    n_weight = 3 if use_value else 2        ## synapses: query, key, value (optional)

    weight_shape = (di, n_weight * dj)
    weights = jnp.zeros(weight_shape)
    w_masks = jnp.zeros(weight_shape)

    for i in range(n_weight): ## W = (Wq, Wk, Wv)
        start_j = i * dj
        end_j = (i + 1) * dj

        weights = weights.at[:, start_j : end_j].set(init_type(patch_shape, subkey[i]))
        w_masks = w_masks.at[:, start_j : end_j].set(jnp.ones(patch_shape))

    return weights, patch_shape, w_masks

def _split_heads(t, n_heads):
    """
    (batch_size, n_patch, d) -> (batch_size, n_heads, n_patch, d / n_heads)
    """
    return t.reshape(t.shape[0], t.shape[1], n_heads, -1).transpose(0, 2, 1, 3)

def _merge_heads(t):
    """
    (batch_size, n_heads, n_patch, d / n_heads) -> (batch_size, n_patch, d)
    """
    return t.transpose(0, 2, 1, 3).reshape(t.shape[0], t.shape[2], -1)

def _attend(inputs, W, b, n_patch, n_heads=1, use_value=True):
    """
    (zF W + b) -> y = softmax(q k.T / sqrt(d_h)) v
    """
    zF = inputs.reshape(inputs.shape[0], n_patch, -1)  ## (B, n, d)
    q_k_v = (zF @ W) + b                               ## q_k_v: (B, n, d)

    if use_value:
        q, k, v = jnp.split(q_k_v, 3, axis=2)          ## q, k v projections of every patch
    else:
        q, k = jnp.split(q_k_v, 2, axis=2)             ## v = zF
        v = zF

    q = _split_heads(q, n_heads)                       ## Reshape: (B, n, d) -> (B, H, n, d/H)
    k = _split_heads(k, n_heads)                       ## Reshape: (B, n, d) -> (_, H, _, d/H)
    v = _split_heads(v, n_heads)                       ## Reshape: (B, n, d) -> (_, H, _, d/H)
    ## attention score
    rd = jnp.sqrt(q.shape[3])
    s = (q @ k.transpose(0, 1, 3, 2)) / rd             ## (B, H, n, d/H) @ (B, H, d/H, n)
    ## probability
    A = nn.softmax(s, axis=-1)
    y = _merge_heads(A @ v)                            ## Reshape.T: (B, H, n, d/H) -> (B, n, d)
    return y.reshape(y.shape[0], -1), (zF, q, k, v, s, A)

def _attend_backward(e, attention_params, n_heads=1):
    """
    d_qkv = (chain rule through softmax(q k.T / sqrt(d_h)) v)
    Passes a post-synaptic signal back through the attention mixing to the (q, k, v) projections of every patch
    """
    zF, q, k, v, s, A = attention_params
    rd = jnp.sqrt(q.shape[3])
    e = _split_heads(e.reshape(e.shape[0], zF.shape[1], -1), n_heads)   ## Reshape: (B, n, d) -> (_, H, _, d/H)
    dv = A.transpose(0, 1, 3, 2) @ e
    dA = e @ v.transpose(0, 1, 3, 2)
    dS = A * (dA - jnp.sum(dA * A, axis=-1, keepdims=True)) / rd
    dq = dS @ k
    dk = dS.transpose(0, 1, 3, 2) @ q
    return jnp.concatenate([_merge_heads(dq),     ## Reshape.T: (B, H, n, d/H) -> (B, n, d)
                            _merge_heads(dk),     ## Reshape.T: (B, H, n, d/H) -> (B, n, d)
                            _merge_heads(dv)],    ## Reshape.T: (B, H, n, d/H) -> (B, n, d)
                           axis=2)

def _backtransmit(e, attention_params, W, n_heads=1, use_value=True):
    """
    pre_out = dq * Wq.T + dk * Wk.T + dv (v = zF)   (chain rule through the weights)
    """
    d_qkv = _attend_backward(e, attention_params, n_heads)
    dzF = d_qkv[:, :, 0:W.shape[1]] @ W.T
    if not use_value: ## (v = zF): the value signal passes straight back
        dzF = dzF + d_qkv[:, :, W.shape[1]:]
    return dzF.reshape(dzF.shape[0], -1)

class AttentionPatchedSynapse(JaxComponent): ## base attention patched synaptic cable
    """
    A patched self-attention synaptic cable that lets the patches (tokens) of a layer attend to one another, i.e., it
    transforms inputs via multi-head self-attention through a single set of query, key, and value synapses that is
    shared by every patch, no form of synaptic evolution/adaptation is in-built to this component.

    | --- Synapse Compartments: ---
    | inputs - input (takes in external signals)
    | outputs - output signals (transformation induced by synapses)
    | project_input - input of the projection pathway (takes in external signals)
    | project_output - output signals of the projection pathway (transformation induced by synapses)
    | post_in - post-synaptic input of the backward pathway (takes in external signals)
    | pre_out - pre-synaptic output signals of the backward pathway (post_in carried back across the synapses)
    | weights - current value matrix of synaptic efficacies, the query, key, and value synapses W = (Wq, Wk, Wv)
    | biases - current value vector of synaptic bias values, b = (bq, bk, bv)
    | key - JAX PRNG key

    Args:
        name: the string name of this cell

        shape: tuple specifying shape of this synaptic cable (usually a 2-tuple
            with number of inputs by number of outputs), every patch gets
            shape[0] / n_patch input and shape[1] / n_patch output
            features (these must be equal if use_value=False)

        n_patch: The number of patches (tokens) in each layer (Default: 1)

        n_heads: The number of attention heads, each head attends over
            (shape[1] / n_patch) / n_heads features (Default: 1)

        use_value: if False, the value synapses are fixed to the identity (v = zF)
            and are left out of `weights`, i.e., W = (Wq, Wk) (Default: True)

        weight_init: a kernel to drive initialization of this synaptic cable's values;
            typically a tuple with 1st element as a string calling the name of
            initialization to use

        bias_init: a kernel to drive initialization of biases for this synaptic cable
            (Default: None, which turns off/disables biases)

        resist_scale: a fixed scaling factor to apply to synaptic transform
            (Default: 1.), i.e., yields: out = attention(in; W, b) * Rscale

        p_conn: probability of a connection existing (default: 1.); setting
            this to < 1. will result in a sparser synaptic structure

        batch_size: the size of each mini batch
    """

    def __init__(self, name, shape, n_patch=1, n_heads=1, use_value=True, weight_init=None, bias_init=None,
            resist_scale=1., p_conn=1., batch_size=1, **kwargs
    ):
        super().__init__(name, **kwargs)

        self.Rscale = resist_scale
        self.batch_size = batch_size
        self.weight_init = weight_init
        self.bias_init = bias_init

        self.n_patch = n_patch
        self.n_heads = n_heads
        self.use_value = use_value

        tmp_key, *subkeys = random.split(self.key.get(), 4)
        if self.weight_init is None:
            info(self.name, "is using default weight initializer!")
            self.weight_init = DistributionGenerator.fan_in_gaussian()

        weights, self.patch_shape, self.w_masks = _create_attention_synapses(
            key=tmp_key, shape=shape, n_patch=self.n_patch, use_value=self.use_value,
            init_type=self.weight_init
        )

        self.shape = shape ## input x output same for all weights (Wq, Wk, Wv)

        if 0. < p_conn < 1.: ## only non-zero and <1 probs allowed
            mask = random.bernoulli(subkeys[1], p=p_conn, shape=weights.shape)
            weights = weights * mask ## sparsify matrix

        ## Compartment setup
        self.weights = Compartment(weights)

        preVals = jnp.zeros((self.batch_size, self.shape[0]))
        postVals = jnp.zeros((self.batch_size, self.shape[1]))

        self.inputs = Compartment(preVals)
        self.outputs = Compartment(postVals)

        self.project_input = Compartment(preVals)
        self.project_output = Compartment(postVals)

        self.post_in = Compartment(postVals)
        self.pre_out = Compartment(preVals)

        ## Set up (optional) bias values
        if self.bias_init is None:
            info(self.name, "is using default bias value of zero (no bias kernel provided)!")
        self.biases = Compartment(self.bias_init((1, weights.shape[1]), subkeys[2]) if bias_init else 0.0)

    @compilable
    def advance_state(self):
        # Get the variables
        weights = self.weights.get()
        biases = self.biases.get()

        ################### inputs  →  attention(W)= A @ v  → outputs
        inputs = self.inputs.get()  ## zF
        ## Compute (zF @ W) -> softmax(q k.T / sqrt(d_h)) v
        outputs, attention_params = _attend(inputs, weights, biases, self.n_patch, self.n_heads, self.use_value)
        _, q, k, v, s, A = attention_params
        outputs = outputs * self.Rscale
        ## Update outputs compartment
        self.outputs.set(outputs)


        ###################    post_in →  (d_qkv @ W.T)  → pre_out
        post_in = self.post_in.get()
        ## Compute attention(post_in, W.T)
        pre_out = _backtransmit(post_in, attention_params, weights, self.n_heads, self.use_value)
        ## Update pre_out compartment
        self.pre_out.set(pre_out)


        ################### project_input →  attention(W)  → project_output
        project_input = self.project_input.get()
        ## Compute attention(project_input, W)
        project_output, _ = _attend(project_input, weights, biases, self.n_patch, self.n_heads, self.use_value)
        project_output = project_output * self.Rscale
        ## Update project_output compartment
        self.project_output.set(project_output)

    @compilable
    def reset(self): ## closed, no-batch argument reset
        ## write reset command to call inner batched_reset command
        self.batched_reset(batch_size=self.batch_size) ## arg = batch_size data-member

    @compilable
    def batched_reset(self, batch_size): ## open, batch argument reset
        preVals = jnp.zeros((batch_size, self.shape[0]))
        postVals = jnp.zeros((batch_size, self.shape[1]))

        # BUG: the self.inputs here does not have the targeted field
        # NOTE: Quick workaround is to check if targeted is in the input or not
        hasattr(self.inputs, "targeted") and not self.inputs.targeted and self.inputs.set(preVals)
        self.outputs.set(postVals)
        self.post_in.set(postVals)
        self.pre_out.set(preVals)
        self.project_input.set(preVals)
        self.project_output.set(postVals)

    @classmethod
    def help(cls): ## component help function
        properties = {
            "synapse_type": "AttentionPatchedSynapse - performs a synaptic transformation "
                            "of inputs to produce output signals (e.g., a multi-head "
                            "self-attention transformation across the patches of a layer)"
        }
        compartment_props = {
            "inputs":
                {"inputs": "Takes in external input signal values",
                 "project_input": "Takes in external input signal values",
                 "post_in": "Takes in external input signal values",
                 },
            "states":
                {"weights": "Synapse efficacy/strength parameter values (query, key, and value synapses)",
                 "biases": "Base-rate/bias parameter values (query, key, and value biases)",
                 "key": "JAX PRNG key"},
            "outputs":
                {"outputs": "Output of synaptic transformation",
                 "project_output": "Output of synaptic transformation",
                 "pre_out": "Output of synaptic transformation",
                 },
        }
        hyperparams = {
            "shape": "Overall shape of synaptic cable, number inputs x number outputs",
            "n_patch": "The number of patches (tokens) in each layer",
            "n_heads": "The number of attention heads",
            "use_value": "Are the value synapses adaptive? (otherwise, they are fixed to the identity)",
            "batch_size": "Batch size dimension of this component",
            "weight_init": "Initialization conditions for synaptic weight (W) values",
            "bias_init": "Initialization conditions for bias/base-rate (b) values",
            "resist_scale": "Resistance level scaling factor (Rscale), applied to output of transformation",
            "w_masks": "weight mask matrix",
            "p_conn": "Probability of a connection existing (otherwise, it is masked to zero)"
        }
        info = {cls.__name__: properties,
                "compartments": compartment_props,
                "dynamics": "outputs = [softmax((inputs * Wq + bq) * (inputs * Wk + bk).T / sqrt(d)) * (inputs * Wv + bv)] * Rscale",
                "hyperparameters": hyperparams}
        return info

if __name__ == '__main__':
    from ngcsimlib.context import Context

    with Context("Bar") as bar:
        Wab = AttentionPatchedSynapse("Wab", (9, 9), 3)
    print(Wab)







