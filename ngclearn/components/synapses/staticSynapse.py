from .denseSynapse import DenseSynapse

class StaticSynapse(DenseSynapse):
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

            weight_init: a kernel to drive initialization of this synaptic cable's values;
                typically a tuple with 1st element as a string calling the name of
                initialization to use

            resist_scale: a fixed (resistance) scaling factor to apply to synaptic
                transform (Default: 1.), i.e., yields: out = ((W * Rscale) * in)

            p_conn: probability of a connection existing (default: 1.); setting
                this to < 1 and > 0. will result in a sparser synaptic structure
                (lower values yield sparse structure)
        """
    pass
