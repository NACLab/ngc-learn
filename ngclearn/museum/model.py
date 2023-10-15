import jax
from jax import numpy as jnp, grad, jit, vmap, random, lax
import os, json
import warnings
from abc import ABC, abstractmethod

## base model class
class Model(ABC):
    """
    Base model class that other (historical) computational models within this
    museum shall inherit from.

    Args:
        name: string name of this computational model

        config: model arguments, e.g., hyper-parameters, experimental settings

        seed: integer key to control/set RNG that drives this node
    """
    def __init__(self, name, config, seed=69):
        self.name = name
        self.config = args
        self.seed = seed
        self.key = random.PRNGKey(seed)

    def save(self, model_directory):
        pass

    def load(self, model_directory):
        pass

#class_name = Model.__name__

################################################################################
