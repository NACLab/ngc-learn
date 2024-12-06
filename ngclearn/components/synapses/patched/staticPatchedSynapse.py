from .patchedSynapse import PatchedSynapse



class StaticPatchedSynapse(PatchedSynapse):
    """
        A static dense synaptic cable; no form of synaptic evolution/adaptation
        is in-built to this component.

        | --- Synapse Compartments: ---
        | inputs - input (takes in external signals)
        | outputs - output
        | weights - current value matrix of synaptic efficacies

        Args:
            name: the string name of this cell

            shape: tuple specifying shape of this synaptic cable (usually a 2-tuple
                with number of inputs by number of outputs)


            n_sub_models: tuple specifying the number of sub models (the number of fully connected/dense synapses)  or the number of patches of this synaptic cable (usually a 2-tuple
                with number of inputs by number of outputs)

            stride_shape: tuple specifying the stride (overlap) between sub models (patches) (Default: (0, 0))

            w_mask= Helps in creating patches (Default: None)

            weight_init: a kernel to drive initialization of this synaptic cable's values;
                typically a tuple with 1st element as a string calling the name of
                initialization to use 

            bias_init: a kernel to drive initialization of this synaptic bias's values;
                typically a tuple with 1st element as a string calling the name of
                initialization to use 

            resist_scale: a fixed (resistance) scaling factor to apply to synaptic
                transform (Default: 1.), i.e., yields: out = ((W * Rscale) * in)

            p_conn: probability of a connection existing (default: 1.); setting
                this to < 1 and > 0. will result in a sparser synaptic structure
                (lower values yield sparse structure)
    """
    pass









