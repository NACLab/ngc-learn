from .sparseTensorSynapse import SparseTensorSynapse

class StaticSparseTensorSynapse(SparseTensorSynapse):
    """"
    A static sparse tensor-synaptic cable; no form of synaptic evolution/adaptation is in-built to this component. Note
    this component cable implements a full, locally-connected structure or an unshared convolutional synaptic
    tensor structural component (cable).

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
    pass