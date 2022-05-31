import tensorflow as tf
import sys
import numpy as np
import copy
from ngclearn.utils import transform_utils

class UpdateRule:
    """
    Base update rule (class from which other rule types inherit basic properties from)

    Args:
        rule_type: the string concretely denoting this rule's type

        name: the string name of this update rule (Default = None which creates an auto-name)
    """
    def __init__(self, rule_type, name=None):
        self.rule_type = rule_type
        self.name = name
        if name is None:
            self.name = "update_rule_{}".format(rule_type)
        self.terms = None
        self.weights = None
        self.cable = None
        self.param_name = None

    def point_to_cable(self, cable, param_name):
        """
        Gives this update rule direct access to the source cable it will update
        (useful for extra statistics often required by certain local synaptic
        adjustment rules).

        Args:
            cable: the cable to point to

            param_name: synaptic parameters w/in this cable to point to
        """
        self.cable = cable
        self.param_name = param_name

    def set_terms(self, terms, weights=None):
        """
        Sets the terms that drive this update rule

        Args:
            terms: list of 2-tuples where each 2-tuple is of the form
                (Node, string_compartment_name)
        """
        self.terms = terms
        self.weights = weights

    def calc_update(self, for_bias=False):
        """
        Calculates the adjustment matrix given this rule's configured internal terms

        Args:
            for_bias: calculate the adjustment vector (instead of a matrix) for a bias

        Returns:
            an adjustment matrix/vector
        """
        pass

    def clone(self):
        pass
