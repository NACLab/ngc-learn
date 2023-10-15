import jax
from jax import numpy as jnp, grad, jit, vmap, random, lax
import os, json
import warnings
from abc import ABC, abstractmethod
from ngclearn.engine.utils.bundle_rules import overwrite

## base cell
class Node(ABC):
    """
    Base node/cell element (class from which other node types inherit basic properties from)

    Args:
        name: string name of this node

        dt: integration time constant, i.e., strength of update to adjust values
            within this node at each simulation step

        seed: integer key to control/set RNG that drives this node
    """
    def __init__(self, name, dt, seed=69):
        self.name = name
        self.seed = seed
        self.key = random.PRNGKey(seed)
        self.dt = dt  # 3. # integration time

        self.config = None
        self.comp = {"in":None, "out": None}
        self.incoming = []  # list of tuples - (callback_function,  dest_comp)
        self.t = 0.  ## node clock
        self.bundle_rules = {None: overwrite(self)}

    def custom_load(self, node_directory):
        return

    def custom_dump(self, node_directory, template=False) -> dict[str, any]:
        return {}

    def dump(self, nodes_directory, overwrite=True, template=False):
        if not os.path.isdir(nodes_directory):
            raise RuntimeError("Can not find directory " + nodes_directory)
        if os.path.isdir(nodes_directory + "/" + self.name):
            if not overwrite:
                raise RuntimeError("Node " + self.name + " already exists in the model")
            else:
                # warnings.warn("Overwriting node " + self.name)
                for filename in os.listdir(nodes_directory + "/" + self.name):
                    file_path = os.path.join(nodes_directory + "/" + self.name, filename)
                    try:
                        if os.path.isfile(file_path) or os.path.islink(file_path):
                            os.unlink(file_path)
                    except:
                        print("failed to remove", file_path)
                        exit()
        else:
            os.mkdir(nodes_directory + "/" + self.name)
        node_directory = nodes_directory + "/" + self.name

        additional_dict = self.custom_dump(node_directory, template=template)

        all_data = {k: self.__dict__.get(k, None) for k in ['name', 'dt', 'seed']} | additional_dict | {'type': self.__class__.__name__}
        with open(node_directory + "/data.json", 'w') as f:
            json.dump(all_data, f)

    def check_compartment(self, comp_name, fatal=False):
        if comp_name not in self.comp.keys():
            if fatal:
                raise RuntimeError("Compartment name " + str(comp_name) + " does not exist")
            else:
                warnings.warn("Compartment name " + str(comp_name) + " does not exist")

    def get(self, comp_name):  ## extract value of neural compartment
        """
        Extracts the data signal value that is currently stored inside of a
        target compartment within this node.

        Args:
            comp_name: name of the compartment in this node to extract data from
        """
        self.check_compartment(comp_name)
        return self.comp.get(comp_name)

    def make_callback(self, comp_name):
        return lambda : self.comp[comp_name]

    def add_cable(self, callback, dest_comp, bundle=None):
        self.incoming.append((callback, dest_comp, bundle))

    def clamp(self, comp_name, signal):
        """
        Clamps an externally provided named value (a vector/matrix) to the desired
        compartment within this node.

        Args:
            comp_name: the (str) name of the compartment to clamp this data signal to

            signal:  the data signal block to clamp to the desired compartment name
        """
        self.check_compartment(comp_name, fatal=True)
        self.comp[comp_name] = signal

    def process_incoming(self):
        for (callback, dest_comp, bundle) in self.incoming:
            yield callback(), dest_comp, bundle

    def pre_gather(self):
        pass

    def add_bundle_rule(self, bundle, rule):
        self.bundle_rules[bundle] = rule

    def gather(self): ## aggregate (transformed) signals from incoming nodes
        self.pre_gather()
        for val, dest_comp, bundle in self.process_incoming():
            self.bundle_rules[bundle](val, dest_comp)

    @abstractmethod
    def step(self):  ## run cell dynamics forward one step
        """
        Executes this node's internal integration/calculation for one
        discrete step in time, i.e., runs simulation of this node for one time step.
        """
        return None

    def set_to_rest(self, batch_size=1, hard=True):
        """
        Wipes/clears values of each compartment in this node.

        Args:
            batch_size: column-dim of this node to reset to (DEFAULT: 1)

                :Note: this is currently unused

            hard: if False, this will NOT clear this node's components upon
                being called (DEFAULT: True)
        """
        if hard:
            self.t = 0.
            ## set compartments to resting states
            for key in self.comp:
                self.comp[key] = None

    @staticmethod
    def get_default_in():
        """
        Returns the value within compartment ``in``
        """
        return 'in'

    @staticmethod
    def get_default_out():
        """
        Returns the value within compartment ``out``
        """
        return 'out'

class_name = Node.__name__

################################################################################
