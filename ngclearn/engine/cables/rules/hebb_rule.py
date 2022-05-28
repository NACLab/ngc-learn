import tensorflow as tf
import sys
import numpy as np
import copy
from ngclearn.utils import transform_utils
from ngclearn.engine.cables.rules.rule import UpdateRule

class HebbRule(UpdateRule):
    """
    The Hebbian update rule.

    Args:
        name: the string name of this update rule (Default = None which creates an auto-name)
    """
    def __init__(self, name=None):
        rule_type = "hebbian"
        super().__init__(rule_type, name)
        self.preact = None
        self.postact = None

    def set_terms(self, terms):
        if len(terms) > 2 or len(terms) <= 0:
            print("ERROR: {} must contain 1 or 2 terms "
                  "(input.len = {}) (rule.name = {})".format(self.rule_type,
                                                             len(terms), self.name))
            sys.exit(1)
        if len(terms) == 2:
            term1 = terms[0]
            self.preact = term1
            term2 = terms[1]
            self.postact = term2
        else:
            term2 = terms[1]
            self.postact = term2

    def calc_update(self, for_bias=False):
        postact_node, postact_comp = self.postact
        postact_term = postact_node.extract(postact_comp)
        if self.preact is not None and for_bias == False: # update matrix
            preact_node, preact_comp = self.preact
            preact_term = preact_node.extract(preact_comp)
            # calculate the final update matrix
            update = tf.matmul(preact_term, postact_term, transpose_a=True)
        else: # vector update
            update = tf.reduce_sum(postact_term, axis=0, keepdims=True)
        return update
