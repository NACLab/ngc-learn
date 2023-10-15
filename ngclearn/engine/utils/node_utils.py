"""
Node global functions and utilities.
"""

def wire_to(cell_1, src_comp, cell_2, dest_comp, bundle=None):
    """
    Connects one node/cell (cell_1) to another node/cell (cell_2), specifically
    wiring the first cell's source compartment to the second cell's destination
    compartment.

    Args:
        cell_1: the first node/cell to build a cable from

        src_comp: source compartment of cell_1 to wire from

        cell_2: the second node/cell to build a cable to

        dest_comp: destination compartment of cell_2 to wire to

        bundle: accompanying bundle to associate with this cable
    """
    cell_2.add_cable(cell_1.make_callback(src_comp), dest_comp, bundle=bundle)
    return ','.join([cell_1.name, src_comp, cell_2.name, dest_comp, str(bundle)])

def wire_hebbian(pre_synaptic, hebbian, post_synaptic, teaching):
    """
    Convenience function for wiring two cells with a cable that implements
    multi-factor Hebbian plasticity.

    Args:
        pre_synaptic: pre-synaptic cell/node to wire from

        hebbian: the synaptic cable with Hebbian plasticity to use

        post_synaptic: post-synaptic cell/node to wire to

        teaching: teaching signal to modulate plasticity
    """
    wires = []
    wires.append(wire_to(pre_synaptic, pre_synaptic.get_default_out(),
                         hebbian, hebbian.get_default_in()))

    wires.append(wire_to(hebbian, hebbian.get_default_out(),
                         post_synaptic, post_synaptic.get_default_in()))

    wires.append(wire_to(pre_synaptic, pre_synaptic.get_default_out(),
                         hebbian, 'pre'))
    wires.append(wire_to(teaching, teaching.get_default_out(),
                         hebbian, 'post'))

    return wires
