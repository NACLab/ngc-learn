"""
Contains all the built-in bundle rules as well as the default one for an unbundled
cable. Bundle(s) usage is to be added to a node's bundle rules
(Note: the overwrite rule is the default rule).
"""

def overwrite(node):
    """
    The default rule for all bundles. Simply overwrites the value in the destination compartment

    Args:
        node: the node construct to add the (default/base-level) overwrite bundle to
    """
    def rule(value, destination_compartment):
        node.comp[destination_compartment] = value
    return rule


def additive(node):
    """
    Adds the value to the current value in the destination compartment

    Args:
        node: the node construct to add the additive bundle to
    """
    def rule(value, destination_compartment):
        node.comp[destination_compartment] += value
    return rule


def windowed(node, window_length):
    """
    Keeps a moving window of the past window length values

    Args:
        node: the node construct to add the window-ed bundle to

        window_length: length of array of (temporal) values to maintain
    """
    def rule(value, destination_compartment):
        if len(node.comp[destination_compartment]) < window_length:
            node.comp[destination_compartment].append(value)
        else:
            node.comp[destination_compartment] = node.comp[destination_compartment][1:] + [value]
    return rule
