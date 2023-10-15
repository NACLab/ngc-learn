#import jax
from ngclearn.engine.utils.node_utils import wire_to
from ngclearn.engine.nodes.node import Node as _Node
import os, warnings
from typing import TypeVar
from ngclearn.engine.nodes.node_factory import Node_Factory
import uuid


Node = TypeVar('Node', bound=_Node)

class OpGraph:
    """
    Implements the full simulation object as a nodes-and-cables system
    (also known as an NGC graph).

    Initializaiton is like so:
    | graph = OpGraph()

    Args:
        args: external arguments to simulation object

            :Note: this is currently unused
    """
    def __init__(self, args=None):
        self.args = args
        self.cycles = []
        self.nodes: dict[str, Node] = {}
        self.rules = []
        self.cables = []
        self.trackers = {} ## goes thru all trackers that exist and calls all their callbacks
        self.tracked_values = {} ## lists of every time you've called track (whatever cbs have dumped); cannot be cleared
        self.factory = Node_Factory()

    """
    XXX

    Args:
        node:

        fatal:
    """
    def check_node(self, node: Node | str, fatal=False):
        name = node.name if isinstance(node, _Node) else node
        if name not in self.nodes.keys():
            if fatal:
                raise RuntimeError("Can not find node " + name + " in graph")
            else:
                warnings.warn("Can not find node " + name + " in graph")

    """
    XXX

    Args:
        nodes:

        fatal:
    """
    def check_nodes(self, nodes, fatal=False):
        for node in nodes:
            self.check_node(node, fatal)

    def add_cycle(self, cycle):
        """
        Adds/sets an execution cycle in this graph

        Args:
            cycle: an ordered list of Node(s) to create an execution cycle for
        """
        self.check_nodes(cycle)
        self.cycles.append(cycle)

    def add_rules(self, rules):
        self.check_nodes(rules)
        self.rules.append(rules)

    def add_nodes(self, node: Node | list[Node]) -> None:
        if isinstance(node, list):
            for n in node:
                self.nodes[n.name] = n
        else:
            self.nodes[node.name] = node

    def probe(self, node_name, comp_name): ## extract a value from the op-graph
        """
        Extract a particular signal from a particular node embedded in this op-graph

        Args:
            node_name: name of the node from the NGC graph to examine

            comp_name: compartment name w/in Node to extract signal from

        Returns:
            an extracted signal (vector/matrix) OR None if either the node does not exist
            or the entire system has not been simulated (meaning that no node dynamics
            have been run yet)
        """
        node = self.nodes.get(node_name)
        if node is None:
            print("ERROR: node {} does not exist!".format(node_name))
        value = node.get(comp_name)
        if value is None:
            print("ERROR: {}.{} does not exist!".format(node_name, comp_name))
        return value

    def clamp(self, node_name, comp_name, value):
        """
        Clamps an externally provided named value (a vector/matrix) to the desired
        compartment within a particular Node of this NGC graph.

        Args:
            node_name: the (str) name of the node to clamp a data signal to.

            comp_name: the (str) name of the node's compartment to clamp this data signal to.

            value: the data signal block to clamp to the desired compartment name
        """
        self.nodes[node_name].clamp(comp_name, value)

    def step(self, x_clamp): ## run one simulation step of the model
        """
        Online function for simulating exactly one discrete time step of this
        simulated op-graph given its exact current state.

        Args:
            x_clamp: list of 3-tuple strings containing named Nodes, their compartments, and values to (persistently)
                clamp on. Note that this list takes on the form:
                [(node1_name, node1_compartment, value), node2_name, node2_compartment, value),...]
        """
        for (node_name, comp_name, value) in x_clamp:
            self.clamp(node_name, comp_name, value)
        for cycle in self.cycles: # run each compute cycle
            for node_i in cycle: # run each node in cycle for one sim step
                node_i.step()

    def evolve(self):
        """
        Calculates the updates to and adjusts any evolvable nodes (such as
        synaptic cables) within this op-graph. Note that this routine, upon
        being called, assumes that plasticity rules to have been embedded to
        this op-graph.
        """
        for rule in self.rules:
            for node_i in rule:
                node_i.evolve()

    def track(self):
        for uid in self.trackers:
            self.tracked_values[uid].append(self.trackers[uid]())

    def set_to_rest(self, batch_size=1, hard=True): ## set all nodes to resting states
        """
        Set all nodes within this op-graph to their respective resting states.
        (Note: this clears/deletes any persistent signals currently embedded w/in
        this graph's Nodes)

        Args:
            batch_size:

            hard: 
        """
        for cycle in self.cycles: # run each compute cycle
            for i in range(len(cycle)): # set each node to rest
                node_i = cycle[i]
                node_i.set_to_rest(batch_size=batch_size, hard=hard)

    def add_cables(self, cables: str | list[str]):
        if isinstance(cables, list):
            for w in cables:
                self.cables.append(w)
        else:
            self.cables.append(cables)

    def dump_graph(self, directory, name="", template=False, persist=[]):
        directory = directory + '/models/' + name
        folders = directory.split('/') + ['nodes']
        prefix = ""
        for folder in folders:
            if not os.path.isdir(prefix + folder):
                os.mkdir(prefix + folder)
            prefix = prefix + folder + "/"

        #Dump the cables to cables.txt
        with open(directory + "/cables.txt", 'w') as f:
            for cable in self.cables:
                f.write(cable + "\n")

        #Dump the cycles to cycles.txt
        with open(directory + "/cycles.txt", 'w') as f:
            for cycle in self.cycles:
                f.write(','.join([c.name for c in cycle]) + "\n")

        #Dump the rules to rules.txt
        with open(directory + "/rules.txt", 'w') as f:
            for rule in self.rules:
                f.write(','.join([r.name for r in rule]) + "\n")

        for node_name in self.nodes:
            self.nodes[node_name].dump(directory + "/nodes",
                template=template if node_name not in persist else False)


    def load_cables(self, path_to_cables):
        with open(path_to_cables + "/cables.txt", 'r') as f:
            cables = [line.rstrip().split(',') for line in f]

        for src_name, src_comp, target_name, target_comp, bundle in cables:
            self.check_node(src_name, fatal=True)
            self.check_node(target_name, fatal=True)

            source_node = self.nodes[src_name]
            target_node = self.nodes[target_name]
            bundle = None if bundle == str(None) else bundle
            self.add_cables(wire_to(source_node, src_comp, target_node, target_comp, bundle))

    def load_cycles(self, path_to_cycles):
        with open(path_to_cycles + "/cycles.txt", 'r') as f:
            cycles = [line.rstrip().split(',') for line in f]
        for cycle in cycles:
            nodes = []
            for node_name in cycle:
                self.check_node(node_name, fatal=True)
                nodes.append(self.nodes[node_name])
            self.add_cycle(nodes)

    def load_rules(self, path_to_rules):
        with open(path_to_rules + "/rules.txt", 'r') as f:
            rules = [line.rstrip().split(',') for line in f]
        for rule in rules:
            nodes = []
            for node_name in rule:
                self.check_node(node_name, fatal=True)
                nodes.append(self.nodes[node_name])
            self.add_rules(nodes)

    def load_nodes(self, path_to_nodes, factory=None, seed=None):
        factory = self.factory if factory is None else factory

        for filename in os.listdir(path_to_nodes + '/nodes'):
            self.add_nodes(factory.make_node(path_to_nodes + "/nodes/" + filename, seed=seed))

    def load(self, path_to_models, name="", seed=None, frozen=False):
        dir = path_to_models + "/models/" + name
        self.load_nodes(dir, seed=seed)
        if not frozen:
            self.load_rules(dir)
        self.load_cycles(dir)
        self.load_cables(dir)

    def make_tracker(self, node_name, compartment):
        self.check_node(node_name, fatal=True)
        cb = self.nodes[node_name].make_callback(compartment)
        uid = uuid.uuid4()
        self.tracked_values[uid] = []
        self.trackers[uid] = cb
        return uid, lambda : self.tracked_values[uid]

    def get_tracked(self, tracker_uid):
        return self.tracked_values[tracker_uid]

    def split_tracker(self, uid, remove_old_tracker=True):
        new_uid = uuid.uuid4()
        self.tracked_values[new_uid] = []
        self.trackers[new_uid] = self.trackers[uid]
        if remove_old_tracker:
            self.remove_tracker(uid)
        return new_uid, lambda : self.tracked_values[new_uid]

    def remove_tracker(self, uid, remove_data=False):
        del self.trackers[uid]
        if remove_data == True:
            del self.tracked_values[uid]

    def clear_tracker(self, uid):
        self.tracked_values[uid].clear()

    def set_optimization(self, opt_algo):
        """
        Sets the internal optimization algorithm used by this simulation object.

        Args:
            opt_algo: optimization algorithm to be used, e.g., SGD, Adam, etc.
                (Note: must be a valid optimizer object.)

                :Note: this is for legacy-support of ngc-learn <0.5.0
        """
        pass
