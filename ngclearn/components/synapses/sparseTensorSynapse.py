from jax import random, numpy as jnp, jit
from functools import partial
import numpy as np

from ngclearn import compilable  # from ngcsimlib.parser import compilable
from ngclearn import Compartment  # from ngcsimlib.compartment import Compartment
from ngclearn.components.synapses import DenseSynapse
#from ngclearn.utils.model_utils import normalize_block_matrix
from ngclearn.utils.distribution_generator import DistributionGenerator as dist


########################################################################################################################
## helper-functions for sparse-synaptic tensors
def _make_connectivity_map(
    P, ## size of sliding input connectivity window
    S, ## number of output streams/maps/groups
    stride=1, ## stream skip (Default: 1)
    padding=0,
    dilation=1,
    convergent_factor=0.0,
    rewire_prob=0.0,
    seed=None,
    total_inputs=None,
):
    """
    This generator produces an (S, P) connectivity map with options for dilated, convergent, and small-world (randomized)
    biological synaptic (tensor/locally-connected) structures.

    Args:
        P: maximum connection window size (defines the tensor shape)
        S: total number of output streams
        stride: stride step between successive output receptive fields
        padding: boundary padding to offset the starting index
        dilation: gap between input indices inside a block (1 = contiguous)
        convergent_factor: if > 0, windows contract symmetrically towards the center of the layer
            (simulating foveation / variable resolution)
        rewire_prob: probability [0, 1] of rewiring any connection to a random shortcut
        seed: random seed for reproducible small-world rewiring
        total_inputs: maximum valid input stream index. Required if using rewire_prob or padding boundaries to
            ensure validity

    Returns:
        a connectivity map for dictating information flow in synaptic locally-connected tensor structure
    """
    ## initialize a map filled with -1 (dead-zone padding marker)
    conn_map = np.full((S, P), -1, dtype=np.int32)
    rng = np.random.default_rng(seed)
    ## if total_inputs isn't specified, estimate it from the basic sliding layout
    if total_inputs is None:
        total_inputs = (S - 1) * stride + (P - 1) * dilation + 1

    for s in range(S):
        ## base starting point for sliding window
        start_idx = s * stride - padding

        ## calculate localized window size if convergent_factor is active; this shrinks window as we move away from layer's center
        if convergent_factor > 0.0:
            center = S / 2.0
            distance_from_center = abs(s - center) / center
            ## shrink window proportionally, ensuring it uses at least 1 connection
            #local_P = max(1, int(P * (1.0 - convergent_factor * distance_from_center)))
            local_P = max(
                1, round(P * (1.0 - convergent_factor * distance_from_center))
            )
        else:
            local_P = P

        ## generate indices incorporating dilation gaps; example: start=0, local_P=3, dilation=2 -> [0, 2, 4]
        indices = start_idx + np.arange(local_P) * dilation

        ## save to map (unfilled slots at tail end remain -1)
        conn_map[s, :local_P] = indices

    ## apply small-world rewiring
    if rewire_prob > 0.0:
        for s in range(S):
            for p in range(P):
                ## only rewire valid existing connections, skipping padding slots (-1)
                if conn_map[s, p] != -1 and rng.random() < rewire_prob:
                    ## swap connection with completely random global input stream
                    conn_map[s, p] = rng.integers(0, total_inputs)
    return jnp.array(conn_map)

@partial(jit, static_argnums=(2, 3, 4, 5))
def _reconstruct_global_2d_matrix(
    weights: jnp.ndarray,
    conn_map: jnp.ndarray,
    total_input_streams: int,
    K_local: int,
    O_local: int,
    S: int,
) -> jnp.ndarray:
    """
    Reconstructs the full 2D block matrix (with zeros) from the dense 4D weight tensor.
    Operates entirely in parallel via JAX JIT, making it highly efficient.

    Args:
        weights: 4D Tensor of shape (P, S, K_local, O_local)
        conn_map: 2D array of shape (S, P) mapping output streams to input streams
        total_input_streams: total unique global input streams (including any padded ones)
        K_local:
        O_local:
        S:

    Returns:
        reconstructed/re-created global block matrix
    """
    P, _, _, _ = weights.shape
    ## Initialize the massive, zero-filled global matrix structure
    ### shape mirrors classical block format: (K_local * total_in, O_local * S)
    global_matrix_h = total_input_streams * K_local
    global_matrix_w = S * O_local
    global_2d_matrix = jnp.zeros((global_matrix_h, global_matrix_w))
    ## to avoid loops, construct coordinate indices for every element in 'weights'
    ### Note: below creates mesh-grids matching dimensions of an individual unshared block
    k_indices, o_indices = jnp.meshgrid(
        jnp.arange(K_local), jnp.arange(O_local), indexing="ij"
    )
    ## broadcast local block dimensions across all S + P positions => expanded shape for coordinate grids: (P, S, K_local, O_local)
    k_grid = jnp.broadcast_to(k_indices[None, None, :, :], weights.shape)
    o_grid = jnp.broadcast_to(o_indices[None, None, :, :], weights.shape)
    ## resolve starting stream-level offsets for every block position;
    ### s_offsets calculates where each output stream block starts horizontally
    s_offsets = jnp.arange(S)[None, :, None, None] * O_local
    ## p_offsets reads conn_map to find where each input block starts vertically
    ### conn_map has shape (S, P) -> transpose to (P, S) to align with weight axes
    p_offsets = jnp.transpose(conn_map, (1, 0))[:, :, None, None] * K_local
    ## compute final absolute 2D global destination coordinates
    global_y_indices = p_offsets + k_grid
    global_x_indices = s_offsets + o_grid
    ## scatter all dense 4D weights into giant 2D matrix in one shot
    global_2d_matrix = global_2d_matrix.at[
        global_y_indices.ravel(), global_x_indices.ravel()
    ].set(weights.ravel())
    return global_2d_matrix

def _make_backwards_connectivity_map(
        forward_conn_map, total_input_streams
):
    ## generates connectivity map + local window size for the backwards pass, safely ignoring -1 boundary padding markers
    S, P = forward_conn_map.shape
    ## initialize tracking dictionary for all valid input stream slots
    in_to_out_links = {i: [] for i in range(total_input_streams)}

    for s in range(S):
        for p in range(P):
            global_in_idx = int(forward_conn_map[s, p])

            ## only map connections that fall inside valid bounds;
            ### this ignores -1 padding placeholders + any indices exceeding total_input_streams
            if 0 <= global_in_idx < total_input_streams:
                in_to_out_links[global_in_idx].append(s)

    ## determine backwards window size (P_back);
    ### if an input has zero connections (e.g. dead boundary edge), default max to 1
    P_back = max(len(links) for links in in_to_out_links.values())
    P_back = max(1, P_back)

    ## build backwards connectivity map of shape (S_back, P_back)
    back_conn_map = np.zeros((total_input_streams, P_back), dtype=np.int32)
    for in_idx, out_indices in in_to_out_links.items():
        ## fill available slots, pad remainder with -1 for safety
        for slot_idx, out_idx in enumerate(out_indices):
            back_conn_map[in_idx, slot_idx] = out_idx
        ## if this input stream connects to fewer outputs than P_back, pad out with -1
        for slot_idx in range(len(out_indices), P_back):
            back_conn_map[in_idx, slot_idx] = -1

    return np.array(back_conn_map), P_back
########################################################################################################################

class SparseTensorSynapse(DenseSynapse):
    """
    A sparse tensor-synaptic cable. Note this component cable implements a full, locally-connected structure or an
    unshared convolutional synaptic tensor structural component (cable).

    | --- Synapse Input Compartments: ---
    | inputs - input (takes in external signals)
    | --- Synapse State Compartments: ---
    | weights - current value matrix of synaptic efficacies (strength values)
    | biases - current value vector of synaptic bias values
    | --- Synapse Output Compartments: ---
    | outputs - output signals

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
            normalize=False,
            norm_axis=1,
            weight_init=None,
            bias_init=None,
            g_conduct_factor=1.,
            p_conn=1.,
            use_block_matrix_format=False,  ## if True, triggers block-matrix implementation
            batch_size=1,
            **kwargs
    ):
        ################################################################################################################
        ## set up internal structure/topology
        ### run geometry math internally first; if invert_conn is True, 'incoming' streams to this component are actually
        ### errors from upper layer, meaning 'in_streams' is upper layer's output streams (S_forward)
        if invert_conn:
            dims = SparseTensorSynapse.initialize_layer_geometry(
                n_in_streams,
                K_local,
                O_local,
                P_l,
                stride,
                auto_pad=False,
                dilation_l=dilation,
                convergent_factor_l=convergent_factor
            )

            total_forward_inputs = n_in_streams
            total_forward_outputs = dims["total_num_output_streams"]

            ## build base forward connectivity map so this can later be "inverted"
            f_map = _make_connectivity_map(
                P_l, S=n_in_streams, stride=stride, convergent_factor=convergent_factor, dilation=dilation
            )
            self.connectivity_map, P_back = _make_backwards_connectivity_map(f_map, total_forward_inputs) ## invert map

            ## explicitly compile exact shapes that ngc-learn expects
            self.n_in_streams = total_forward_outputs  ## e.g., 1 (number of incoming error maps)
            self.n_out_streams = total_forward_inputs  ## e.g., 3 (number of feedback target maps)
            self.K_local = O_local  # Error size (9)
            self.O_local = K_local  # Hidden feature target size (16)

            # Final 4D tensor shape for backwards weights
            calculated_shape = (P_back, total_forward_inputs, O_local, K_local)
            self.io_shape = calculated_io_shape = (self.n_in_streams * self.K_local, self.n_out_streams * self.O_local)

            # Clear window strides for pure backward routing execution
            self.stride = 0
            self.dilation = 1
        else:
            ## standard forward-projection synaptic cable
            dims = SparseTensorSynapse.initialize_layer_geometry(
                n_in_streams,
                K_local,
                O_local,
                P_l,
                stride,
                auto_pad=False,
                dilation_l=dilation,
                convergent_factor_l=convergent_factor
            )

            self.n_in_streams = n_in_streams # dims["total_input_streams"]  # nlm1_streams
            self.n_out_streams = dims["total_num_output_streams"]  # Computed S
            self.K_local = K_local
            self.O_local = O_local

            self.connectivity_map = _make_connectivity_map(
                P_l, dims["total_num_output_streams"], stride, convergent_factor=convergent_factor, dilation=dilation
            )

            calculated_shape = dims["forward_weight_shape"]  # (P_l, S, K_local, O_local)
            self.io_shape = calculated_io_shape = (dims["global_input_dim"], dims["global_output_dim"])
            self.stride = stride
            self.dilation = dilation
        ################################################################################################################
        super().__init__(  ## now, call parent constructor using internally calculated shapes
            name,
            shape=calculated_shape,
            weight_init=weight_init,
            bias_init=bias_init,
            g_conduct_factor=g_conduct_factor,
            p_conn=p_conn,
            batch_size=batch_size,
            **kwargs
        )

        ## store framework properties
        self.convergent_factor = convergent_factor
        self.invert_conn = invert_conn
        self.use_block_matrix_format = use_block_matrix_format
        self.Rscale = g_conduct_factor
        self.normalize = normalize
        self.norm_axis = norm_axis

        #if self.use_effective_dim_prior:
        self.norm_trigger = Compartment(jnp.zeros((1, 1)))
        if self.normalize:
            self.norm_trigger.set(self.norm_trigger.get() + 1)
            P, S, Klocal, Olocal = self.shape
            norm_axis = 1 #0 #1
            Weffdim = Klocal * P
            if norm_axis == 1:
                Weffdim = Olocal * self.n_out_streams #Weffdim = Olocal
            weight_init = dist.gaussian(mean=0., std=float(1.0 / jnp.sqrt(Weffdim)))
            weights = weight_init(self.shape, self.key.get())
            self.weights.set(weights)

        if self.use_block_matrix_format: ## set up block-diagonal algorithm backend if flagged
            block_weights = _reconstruct_global_2d_matrix(
                self.weights.get(),
                self.connectivity_map,
                self.n_in_streams, ## total num in-streams
                K_local=self.shape[2],
                O_local=self.shape[3],
                S=self.n_out_streams #self.shape[1] ## total num out-streams
            )
            self.weights.set(block_weights)

        if self.normalize: ## pre-normalize synapses if norm-constraints are to be used
            if not self.use_block_matrix_format:
                weights = SparseTensorSynapse._normalize_outgoing_weights(self.weights.get())
            else:
                weights = SparseTensorSynapse._normalize_block_matrix_outgoing_weights(
                    self.weights.get(),
                    self.connectivity_map,
                    self.n_in_streams,
                    K_local,
                    O_local,
                    self.n_out_streams,
                    axis=1, #0, #1,
                    order=2,
                    norm_targ=1.
                )
            self.weights.set(weights)


        self.initial_weights = Compartment(self.weights.get())  ## store synaptic initial conditions (never updated)
        preVals = jnp.zeros((self.batch_size, self.io_shape[0]))
        postVals = jnp.zeros((self.batch_size, self.io_shape[1]))
        self.inputs.set(preVals)
        self.outputs.set(postVals)

    @staticmethod
    def initialize_layer_geometry( ## internal co-routine for tensor-synaptic projection shaping
            nlm1_streams,
            Nlm1,
            Nl,
            P_l,
            stride_l,
            auto_pad=False,
            dilation_l=1,
            convergent_factor_l=0.0,
    ):
        ## this sets up geometrical constraints for this synaptic tensor component, notably this:
        ### * calculates output stream count (S), maximum structural input reach, and
        ### * allocates flat sizes for ngc-learn buffers in one unified sweep

        ## check 1: dilation stride minimum bounds
        if stride_l < 1:
            raise ValueError(
                "Geometry Error: stride_l must be greater than or equal to 1."
            )
        ## check 2: check physical window limits if padding is disabled
        max_physical_span = (P_l - 1) * dilation_l + 1
        if max_physical_span > nlm1_streams and not auto_pad:
            raise ValueError(
                f"Geometry Error: Receptive field physical span ({max_physical_span} inputs) "
                f"exceeds incoming streams ({nlm1_streams}). Enable auto_pad."
            )

        ## sweep over potential output streams to find exact boundary
        s = 0
        max_seen_input_idx = -1
        while True:
            start_idx = s * stride_l
            ## under foveation/convergence, calculate local window footprint dynamically
            ### we estimate center distance dynamically as we look ahead
            if convergent_factor_l > 0.0:
                ## we use a running estimation of center scaling
                ## for a self-contained lookup, we approximate shrinking footprint relative to stride
                estimated_S_guess = max(1, nlm1_streams // stride_l)
                center = estimated_S_guess / 2.0
                distance_from_center = abs(s - center) / center
                local_P = max(
                    1, int(P_l * (1.0 - convergent_factor_l * distance_from_center))
                )
            else:
                local_P = P_l
            ## compute absolute furthest index this specific stream row will touch
            last_index_in_row = start_idx + (local_P - 1) * dilation_l

            if auto_pad: ## boundary evaluation gates
                ## stop if starting position completely falls off available canvas
                if start_idx >= nlm1_streams:
                    break
            else:
                ## without padding, entire synaptic projection footprint must fit w/in source bounds
                if last_index_in_row >= nlm1_streams:
                    break
            ## if stream is valid, track its absolute furthest input coordinate reach
            if last_index_in_row > max_seen_input_idx:
                max_seen_input_idx = last_index_in_row
            s += 1

        ## finalize verified number of output streams (S)
        nl_streams = max(1, s)
        ## calculate unique input streams required to fully span generated matrix/tensor
        total_input_streams = max(1, max_seen_input_idx + 1)
        ## calculate how many extra virtual padding streams are absorbed by "dead zone"
        pad_streams = max(0, total_input_streams - nlm1_streams)
        ## build final (structural) metadata values
        forward_weight_shape = (P_l, nl_streams, Nlm1, Nl)
        return {
            "total_num_output_streams": nl_streams,  ## computed num output streams (S)
            "pad_streams": pad_streams,  ## num streams absorbed by padding filters
            "total_num_input_streams": total_input_streams, ## total span across input map
            "global_input_dim": nlm1_streams * Nlm1,  ## raw size of incoming layer nodes
            "global_output_dim": nl_streams * Nl,  ## raw size of output nodes
            "forward_weight_shape": forward_weight_shape, ## locally-connected tensor shape
        }

    @compilable
    def advance_state(self):
        weights = self.weights.get()  ## get synaptic tensor
        biases = self.biases.get()
        inputs = self.inputs.get()  ## get inputs
        conn_map = self.connectivity_map  ## get connectivity structure
        #print(self.name, " ", weights.shape, " ", inputs.shape)

        B = inputs.shape[0]  ## get batch size for subsequent averaging
        if not self.use_block_matrix_format:
            P, S, K_local, O_local = weights.shape
            ## reshape global inputs into standard patches
            patched_inputs = inputs.reshape(B, -1, K_local)
            total_input_streams = patched_inputs.shape[1]

            ## append an explicit row of absolute zeros to the end of our inputs;
            ### this acts as dead-zone buffer for all invalid/padded indices
            zero_padding_block = jnp.zeros((B, 1, K_local))
            padded_inputs = jnp.concatenate(
                [patched_inputs, zero_padding_block], axis=1
            )

            ## create a safe connectivity map by replacing all -1 flags or
            ### out-of-bounds indices with a pointer to our dead-zone block;
            ### (dead-zone block sits at very last index: total_input_streams)
            is_invalid = (conn_map == -1) | (conn_map < 0) | (conn_map >= total_input_streams)
            safe_conn_map = jnp.where(is_invalid, total_input_streams, conn_map)

            ## gather overlapping streams using our safe map => (B, S, P, K_local)
            ### invalid indices safely pull from dead-zone block of zeros
            gathered_inputs = padded_inputs[:, safe_conn_map, :]
            ## transpose to move P to front: (P, B, S, K_local)
            gathered_inputs = jnp.transpose(gathered_inputs, (2, 0, 1, 3))

            ## use None axes to broadcast (P, B, S, K_local, 1) * (P, 1, S, K_local, O_local)
            ### output of this elementwise multiply is shape: (P, B, S, K_local, O_local)
            elementwise_prod = (
                gathered_inputs[..., None] * weights[:, None, :, :, :]
            )

            ## sum out the input stream dimensions: axis 0 (P) and axis 3 (K_local)
            ### resulting shape: (B, S, O_local)
            outputs_3d = jnp.sum(elementwise_prod, axis=(0, 3))
            outputs = outputs_3d.reshape(B, -1)  ## flatten to 2D formatted output
        else:
            outputs = inputs @ weights  ## block-diagonal/block-matrix multiply
        outputs = (outputs * self.g_conduct_factor) + biases
        self.outputs.set(outputs)

    @compilable
    def reset(self):  ## reset compartments/statistics
        if not self.inputs.targeted:
            self.inputs.set(self.inputs.get() * 0)
        self.outputs.set(self.outputs.get() * 0)  # outputs

    @staticmethod
    def _normalize_outgoing_weights( ## internal tensor normalizer co-routine
            weights: jnp.ndarray,
            axis: int = 1,  # 0,
            order: int = 2,
            epsilon: float = 1e-8
    ) -> jnp.ndarray:
        ## normalizes outgoing weights of each input neuron to a unit L2 norm;
        ### weight tensor shape: (P, S, K_local, O_local)
        ### - axis 0 (P) and Axis 2 (K_local) isolate an individual input neuron.
        ### - axis 1 (S) and Axis 3 (O_local) represent all the places its synapses land.

        if axis == 0:
            ## input-wise normalization: reduce across patch connections (P) and local features (K_local)
            reduce_axes = (0, 2)
        elif axis == 1:
            ## output-wise normalization: reduce across output streams (S) and local output neurons (O_local)
            reduce_axes = (1, 3)
        else:
            raise ValueError("Norm.axis must be 0 or 1.")

        if order == 1:
            ## compute L1 norm across all target output streams (S) and output units (O_local);
            outgoing_norms = jnp.sum(
                jnp.abs(weights),
                axis=reduce_axes,
                # axis=(0, 2), ## equiv to axis=0 for 2D synapse projection
                # axis=(1, 3),  ## equiv to axis=1 for 2D synapse projection
                keepdims=True  ## keepdims=True ensures shape matches for broadcasted division
            )
        else: # order==2
            ## compute L2 norm across all target output streams (S) and output units (O_local);
            outgoing_norms = jnp.sqrt(
                jnp.sum(
                    jnp.square(weights),
                    axis=reduce_axes,
                    # axis=(0, 2), ## equiv to axis=0 for 2D synapse projection
                    #axis=(1, 3),  ## equiv to axis=1 for 2D synapse projection
                    keepdims=True ## keepdims=True ensures shape matches for broadcasted division
                )
            )
        ## divide elementwise to ensure sum of squared outgoing weights for any input neuron equals exactly 1.0.
        return weights / (outgoing_norms + epsilon)


    @staticmethod  #@partial(jit, static_argnums=(2, 3, 4, 5, 6, 7, 8, 9))
    def _normalize_block_matrix_outgoing_weights( ## internal block-matrix normalizer co-routine
            global_matrix: jnp.ndarray,
            conn_map: jnp.ndarray,
            total_input_streams: int,
            K_local: int,
            O_local: int,
            S: int,
            axis: int = 1, #0,
            order: int = 2,
            norm_targ: float = 1.0,
            eps: float = 1e-8,
    ) -> jnp.ndarray:
        ## normalizes only active blocks within an arbitrary overlapping 2D block matrix;
        ### NOTE: leaves all structural zero blocks completely untouched
        P = conn_map.shape[1]
        ## generate the exact coordinate grids for the active weights
        ### this matches the coordinate mapping logic from our reconstruction code
        k_indices, o_indices = jnp.meshgrid(
            jnp.arange(K_local), jnp.arange(O_local), indexing="ij"
        )

        k_grid = jnp.broadcast_to(k_indices[None, None, :, :], (P, S, K_local, O_local))
        o_grid = jnp.broadcast_to(o_indices[None, None, :, :], (P, S, K_local, O_local))
        s_offsets = jnp.arange(S)[None, :, None, None] * O_local
        p_offsets = jnp.transpose(conn_map, (1, 0))[:, :, None, None] * K_local

        global_y_indices = p_offsets + k_grid
        global_x_indices = s_offsets + o_grid
        ## pull out ONLY the valid weights from the 2D matrix into a 4D tensor
        ### gathered_weights shape: (P, S, K_local, O_local)
        gathered_weights = global_matrix[global_y_indices, global_x_indices]

        ## compute norms on the isolated valid blocks based on the desired axis
        if axis == 0:
            ## input-wise normalization: reduce across patch connections (P) and local features (K_local)
            reduce_axes = (0, 2)
        elif axis == 1:
            ## output-wise normalization: reduce across output streams (S) and local output neurons (O_local)
            reduce_axes = (1, 3)
        else:
            raise ValueError("Norm.axis must be 0 or 1.")

        norms = jnp.linalg.norm(
            gathered_weights, ord=order, axis=reduce_axes, keepdims=True
        )
        normalized_gathered = gathered_weights * (norm_targ / (norms + eps)) ## normalize the isolated valid blocks

        ## initialize a pristine zero matrix and scatter the normalized weights back into it
        ### this guarantees that all structural zeros remain mathematically clean 0.0s
        global_matrix_h = total_input_streams * K_local
        global_matrix_w = S * O_local
        new_global_matrix = jnp.zeros((global_matrix_h, global_matrix_w))

        new_global_matrix = new_global_matrix.at[
            global_y_indices.ravel(), global_x_indices.ravel()
        ].set(normalized_gathered.ravel())
        return new_global_matrix

