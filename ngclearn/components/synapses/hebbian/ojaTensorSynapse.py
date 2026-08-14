from jax import random, numpy as jnp, jit
#from functools import partial
from ngclearn import compilable
from ngclearn import Compartment
from ngclearn.components.synapses import SparseTensorSynapse
from ngclearn.utils.distribution_generator import DistributionGenerator as dist

class OjaTensorSynapse(SparseTensorSynapse):
    """
    A sparse tensor-synaptic cable that is adapted via Hebbian or Oja-plasticity. Note this component cable implements
    a full, locally-connected structure or an unshared convolutional synaptic tensor structural component (cable).

    | --- Synapse Input Compartments: ---
    | inputs - input (takes in external signals)
    | --- Synapse State Compartments: ---
    | weights - current value matrix of synaptic efficacies (i.e., the strength values)
    | biases - current value vector of synaptic bias values
    | --- Synapse Output Compartments: ---
    | outputs - output signals

    | References:
    | Oja, E., 1982. Simplified neuron model as a principal component analyzer. Journal of mathematical biology,
    | 15(3), pp.267-273.

    Args:
        name: the string name of this cell

        n_in_streams: total number of incoming streams

        K_local: input local block feature size

        O_local: output local block feature size (note: total number of output streams is computed internally)

        P_l: local window size (how many input streams to grab), i.e., maximum connection window size
            (defines tensor shape) (Default: 1)

        stride: stride factor for locally-connected structure to "skip over" (in terms of incoming streams);
            stride step between successive output receptive field streams (Default: 1)

        convergent_factor: if > 0, windows contract symmetrically towards center of layer
            (simulates foveation or variable resolution) (Default: 0)

        dilation: dilation factor; gap between input indices inside a block (1 = contiguous) (Default: 1)

        invert_conn: if True, this tensor shape will internally "transpose" itself to  formulate the appropriate
            locally-connected inverted/transposed structure (note that this means the shape will no longer be what
            the constructor's argument dictate - it will be the effective transpose of these arguments) (Default: False)

        eta: global learning rate

        oja_factor: if > 0, this triggers the Oja-correction factor to Hebbian plasticity (Default: 0)

        sign_value: multiplicative factor to apply to final synaptic update before it is applied to synapses; this is
            useful if gradient descent style optimization is required (as Hebbian rules typically yield adjustments
            for ascent)

        normalize: if True, this synaptic tensor will normalize its internal blocks (Default: False)

        norm_axis: axis upon which block norms are computed (Default: 1)

        weight_init: a kernel to drive initialization of this synaptic cable's values;
            typically a tuple with 1st element as a string calling the name of
            initialization to use

        bias_init: a kernel to drive initialization of biases for this synaptic cable
            (Default: None, which turns off/disables biases)

        g_conduct_factor: a fixed (resistance) scaling factor to apply to synaptic
            transform (Default: 1.), i.e., yields: out = ((W * in) * g_conduct_factor) + bias

        p_conn: probability of a connection existing (default: 1.); setting this to < 1 and > 0. will result in a
            sparser synaptic structure (lower values yield sparse structure)

        use_block_matrix_format: if True, this tensor synapse resorts to a memory-intensive block-matrix
            computational format (Default: False)

    """

    def __init__(
            self,
            name,
            n_in_streams,  ## total incoming streams (nlm1_streams)
            K_local,  ## input local block feature size
            O_local,  ## output local block feature size
            P_l=1,  ## local window size (how many input streams to grab)
            stride=1,
            convergent_factor=0.,
            dilation=1,
            invert_conn=False,
            eta=0.,
            oja_factor=1.,  ## if <= 0, will disable Oja's correction term
            sign_value=1.,
            normalize=False,
            norm_axis=1,
            weight_init=None,
            bias_init=None,
            p_conn=1.,
            resist_scale=1.,
            batch_size=1,
            use_block_matrix_format=False,  ## if True, triggers block-matrix implementation
            **kwargs
    ):
        super().__init__(
            name,
            n_in_streams=n_in_streams,  # total incoming streams (nlm1_streams)
            K_local=K_local,  ## input local block feature size
            O_local=O_local,  ## output local block feature size
            P_l=P_l,  ## local window size (how many input streams to grab?)
            stride=stride,
            convergent_factor=convergent_factor,
            dilation=dilation,
            invert_conn=invert_conn,
            normalize=normalize,
            norm_axis=norm_axis,
            use_block_matrix_format=use_block_matrix_format,
            weight_init=weight_init,
            bias_init=bias_init,
            resist_scale=resist_scale,
            p_conn=p_conn,
            batch_size=batch_size,
            **kwargs
        )

        ## store cable framework properties
        self.eta = eta
        self.sign_value = sign_value
        self.oja_factor = oja_factor

        # if self.normalize:
        #     P, S, Klocal, Olocal = self.shape
        #     norm_axis = 1 #0 #1
        #     Weffdim = Klocal * P
        #     if norm_axis == 1:
        #         Weffdim = Olocal * self.n_out_streams #Weffdim = Olocal
        #     weight_init = dist.gaussian(mean=0., std=float(1.0 / jnp.sqrt(Weffdim)))
        #     weights = weight_init(self.shape, self.key.get())
        #     self.weights.set(weights)

        preVals = jnp.zeros((self.batch_size, self.io_shape[0]))
        postVals = jnp.zeros((self.batch_size, self.io_shape[1]))
        self.pre = Compartment(preVals)
        self.post = Compartment(postVals)
        self.dWeights = Compartment(self.weights.get() * 0)
        #self.dBiases = Compartment(jnp.zeros(shape[1]))
        self.variance_term = Compartment(postVals)

    @compilable
    def evolve(self, dt):
        ## get compartment variables
        pre = self.pre.get()
        post = self.post.get()
        weights = self.weights.get()
        # biases = self.biases.get()
        key, *subkeys = random.split(self.key.get(), 3)  ## generate keys for noise samples
        self.key.set(key)

        B = pre.shape[0]
        if not self.use_block_matrix_format: ## use tensor format
            conn_map = self.connectivity_map  ## get connectivity structure
            inputs = pre
            outputs = post #+ var_term
            P, S, K_local, O_local = weights.shape

            ## 1st term - calculate (Hebbian) correlation term
            ## gather inputs and transpose to: (P, B, S, K_local)
            patched_inputs = inputs.reshape(B, -1, K_local)
            pre = jnp.transpose(patched_inputs[:, conn_map, :], (2, 0, 1, 3))
            ## reshape outputs to structure: (B, S, O_local)
            post = outputs.reshape(B, S, O_local)
            ## use None axes to broadcast (P, B, S, K_local, 1) * (1, B, S, 1, O_local)
            ### resulting shape: (P, B, S, K_local, O_local)
            outer_product_grid = pre[..., None] * post[None, :, :, None, :]
            ## sum over the batch dimension (axis 1) -> (P, S, K_local, O_local)
            dWeights_hebb = jnp.sum(outer_product_grid, axis=1)

            ## 2nd term - apply (optional) Oja's correction factor
            dWeights_oja = 0.
            if self.oja_factor > 0.:
                ## compute squared postsynaptic activation; (B, S, O_local)
                post_squared = jnp.square(post)
                ## average across batch dimension to get expected decay weight per neuron; (S, O_local)
                mean_post_squared = jnp.sum(post_squared, axis=0)
                ## broadcast decay against current synaptic weights
                ### (1, S, 1, O_local) * (P, S, K_local, O_local) -> (P, S, K_local, O_local)
                dWeights_oja = -(mean_post_squared[None, :, None, :] * weights)

            dWeights = dWeights_hebb + dWeights_oja
        else: ## use block-matrix format (memory-intensive)
            dWeights_hebb = pre.T @ post  ## classical Hebbian term
            dWeights_oja = 0.
            if self.oja_factor > 0.: ### 2nd, apply Oja's (corrective) factor
                dWeights_oja = -(post[:, None] ** 2 * weights)  ## Oja-correction / stability term
                dWeights_oja = jnp.sum(dWeights_oja, axis=0)  ## calculate full (tensor) Oja's correction
            dWeights = (dWeights_hebb + dWeights_oja) * (weights != 0.)  ## calculate full (tensor) update

        weights = weights + dWeights * self.sign_value

        ## apply post-projection / normalization steps
        if self.norm_trigger.get() > 0.:
            if not self.use_block_matrix_format:
                weights = SparseTensorSynapse._normalize_outgoing_weights(weights)
            else:
                weights = SparseTensorSynapse._normalize_block_matrix_outgoing_weights(
                    weights,
                    self.connectivity_map,
                    self.n_in_streams,
                    self.K_local,
                    self.O_local,
                    self.n_out_streams,
                    axis=1, #0, # 1
                    order=2,
                    norm_targ=1.
                )

        ## update compartments
        self.weights.set(weights)
        self.dWeights.set(dWeights)

    @compilable
    def reset(self):  ## reset core components/statistics
        if not self.inputs.targeted:
            self.inputs.set(self.inputs.get() * 0)
        self.outputs.set(self.outputs.get() * 0)  # outputs
        self.pre.set(self.pre.get() * 0)  # pre
        self.post.set(self.post.get() * 0)  # post
        self.dWeights.set(self.dWeights.get() * 0)  # dW
        #self.dBiases.set(self.dBiases.get() * 0)  # db
        self.variance_term.set(self.variance_term.get() * 0)

