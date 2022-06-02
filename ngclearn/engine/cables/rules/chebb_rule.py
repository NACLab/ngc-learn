import tensorflow as tf
import sys
import numpy as np
import copy
from ngclearn.utils import transform_utils as transform
from ngclearn.engine.cables.rules.rule import UpdateRule

class CHebbRule(UpdateRule):
    """
    The contrastive, bounded Hebbian update rule. Note that this rule, when
    used in tandem with spiking nodes and variable traces, also implements the
    online spike-timing dependent plasticity (STDP) rule.

    Args:
        name: the string name of this update rule (Default = None which creates an auto-name)
    """
    def __init__(self, name=None):
        rule_type = "contrastive_hebbian"
        super().__init__(rule_type, name)
        # soft bound coefficient terms
        self.w_min = 0.0
        self.eta_minus = 1 #0.5
        self.w_max = 5.0
        self.eta_plus = 1 #0.5
        self.use_hard_bound = False

    def clone(self):
        rule = CHebbRule(self.name)
        rule.terms = self.terms
        rule.cable = self.cable
        rule.param_name = self.param_name
        rule.w_min = self.w_min
        rule.eta_minus = self.eta_minus
        rule.w_max = self.w_max
        rule.eta_plus = self.eta_plus
        rule.use_hard_bound = self.use_hard_bound
        return rule

    def set_terms(self, terms, weights=None):
        if len(terms) == 4:
            self.terms = terms
            self.weights = weights
        else:
            print("ERROR: {} must contain 4 terms "
                  "(input.len = {}) (rule.name = {})".format(self.rule_type,
                                                             len(terms), self.name))
            sys.exit(1)

    def calc_update(self, for_bias=False):
        w0 = 1
        w1 = 1
        if self.weights is not None:
            w0 = self.weights[0]
            w1 = self.weights[1]

        preact1 = self.terms[0]
        postact1 = self.terms[1]
        preact2 = self.terms[2]
        postact2 = self.terms[3]

        preact_node1, preact_comp1 = preact1
        #preact_tar1 = preact_node1.extract("x_tar")
        preact_term1 = preact_node1.extract(preact_comp1)
        postact_node1, postact_comp1 = postact1
        postact_term1 = postact_node1.extract(postact_comp1)

        preact_node2, preact_comp2 = preact2
        preact_term2 = preact_node2.extract(preact_comp2)
        postact_node2, postact_comp2 = postact2
        #postact_tar2 = postact_node2.extract("x_tar")
        postact_term2 = postact_node2.extract(postact_comp2)

        A_plus = 1.0
        A_minus = 1.0
        if self.cable is not None:
            params = self.cable.params[self.param_name]
            if self.use_hard_bound == False: # soft bounds
                # A_+ = (w_max - w_j) * eta_+
                A_plus = (self.w_max - params) #* self.eta_plus
                # A_- = (w_j - w_min) * eta_-
                A_minus = (params - self.w_min) #* self.eta_minus
            else: # hard bounds
                # A_+ = Heaviside(w_max - w_j) * eta_+
                A_plus = transform.gte(self.w_max - params) #* self.eta_plus
                # A_- = Heaviside(w_j - w_min) * eta_-
                A_minus = transform.gte(params - self.w_min) #* self.eta_minus

        if for_bias == False: # update matrix
            delta_plus = (tf.matmul(preact_term1 * w0, postact_term1 * w1, transpose_a=True)) * A_plus
            delta_minus = tf.matmul(preact_term2 * w0, postact_term2 * w1, transpose_a=True) * A_minus
            # calculate the final update matrix
            #update = delta_plus + delta_minus
            update = delta_plus * self.eta_plus - delta_minus * self.eta_minus
            #update = delta_plus * self.eta_plus + delta_minus * self.eta_minus
            #update = delta_minus * self.eta_minus
            #update = delta_plus * self.eta_plus
        else: # vector update
            delta_plus = tf.reduce_sum(postact_term1 * w1, axis=0, keepdims=True) * A_plus
            delta_minus = tf.reduce_sum(postact_term2 * w1, axis=0, keepdims=True) * A_minus
            update = delta_plus * self.eta_plus - delta_minus * self.eta_minus
        return update
