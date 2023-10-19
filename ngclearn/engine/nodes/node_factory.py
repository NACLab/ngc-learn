import warnings, json
from os.path import join
import glob
import importlib
import sys
import jax

class Node_Factory:
    def __init__(self):
        self.node_types = {}
        self.load_from_dir('ngclearn/engine/nodes/cells')
        self.load_from_dir('ngclearn/engine/nodes/ops')
        self.load_from_dir('ngclearn/engine/nodes/synapses')

    def load_from_dir(self, path):
        modules = glob.glob(join(sys.path[1] + '/' + path, "*.py"))

        for module in modules:
            if module.endswith('__init__.py'):
                continue
            mod = importlib.import_module('.' + module.split('/')[-1][:-3], path.replace('/', '.'))
            try:
                self.add_node_type(getattr(mod, mod.class_name))
            except:
                warnings.warn("Unable to load module " + module)

    def add_node_type(self, nodeClass):
        identifier = nodeClass.__name__
        if identifier in self.node_types:
            warnings.warn("Node class " + identifier + " already exists in the node factory")
        else:
            self.node_types[identifier] = nodeClass

    def make_node(self, directory):
        with open(directory + "/data.json", 'r') as f:
            data = json.load(f)

        node_class = self.node_types[data['type']]
        del data['type']
        if data['key'] is not None:
            key = jax.numpy.array(data['key'], dtype=jax.numpy.uint32)
            data['key'] = key

        node = node_class(**data)
        node.custom_load(directory)
        return node
