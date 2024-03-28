from abc import ABC, abstractmethod
import warnings
import ngclib.utils as utils


class _VerboseDict(dict):
    """
    The Verbose Dictionary functions like a traditional python dictionary with
    more specific warnings and errors.
    Specifically each verbose dictionary logs when new keys are added to it,
    and when a key is asked for that is not present in the dictionary, it will
    throw a runtime error that includes the name (retrieved from an ngclib
    component) to make debugging/tracing easier.

    Args:
        seq: sequence of items to add to the verbose dictionary

        name: the string name of this verbose dictionary

        kwargs: the keyword arguments to first try to extract from
    """

    def __init__(self, seq=None, name=None, verboseDict_showSet=False, **kwargs):
        seq = {} if seq is None else seq
        name = 'unnamed' if name is None else str(name)

        super().__init__(seq, **kwargs)
        self.name = name
        self.showSet = verboseDict_showSet

    def __setitem__(self, key, value):
        if self.showSet and key not in self.keys():
            warnings.warn("Adding key \"" + str(key) + "\" to " + self.name)
        super().__setitem__(key, value)

    def __getitem__(self, item):
        if item not in self.keys():
            raise RuntimeError("Failed to find compartment \"" + str(item) + "\" in " + self.name +
                               "\nAvailable compartments " + str(self.keys()))
        return super().__getitem__(item)


class _ComponentMetadata:
    """
    Component Metadata is used to track the incoming and outgoing connections
    to a component. This is also where all of the root calls exist for verifying
    that components have the correct number of connections.

    Args:
        name: the string name of this component metadata object
    """

    def __init__(self, name, **kwargs):
        self.component_name = name
        self._incoming_connections = {}
        self._outgoing_connections = {}

    def add_outgoing_connection(self, compartment_name):
        """
        Adds an outgoing connection ("cable") to this component to a compartment
        in a node/component elsewhere.

        Args:
            compartment_name: compartment target in an outgoing component/node
        """
        if compartment_name not in self._outgoing_connections.keys():
            self._outgoing_connections[compartment_name] = 1
        else:
            self._outgoing_connections[compartment_name] += 1

    def add_incoming_connection(self, compartment_name):
        """
        Adds an incoming connection ("cable") to this component from a compartment
        in a node/component elsewhere.

        Args:
            compartment_name: compartment source in an incoming component/node
        """
        if compartment_name not in self._incoming_connections.keys():
            self._incoming_connections[compartment_name] = 1
        else:
            self._incoming_connections[compartment_name] += 1

    def check_incoming_connections(self, compartment, min_connections=None, max_connections=None):
        """
        Checks/validates the incoming information source structure/flow into this component.

        Args:
            compartment: compartment from incoming source to check

            min_connections: minimum number of incoming connections this component should receive

            max_connections: maximum number of incoming connections this component should receive
        """
        if compartment not in self._incoming_connections.keys() and min_connections is not None:
            raise RuntimeError(
                str(self.component_name) + " has an incorrect number of incoming connections.\nMinimum connections: " +
                str(min_connections) + "\t Actual Connections: None")

        if compartment in self._incoming_connections.keys():
            count = self._incoming_connections[compartment]
            if min_connections is not None and count < min_connections:
                raise RuntimeError(
                    str(self.component_name) + "has an incorrect number of incoming connections.\nMinimum "
                                               "connections: " +
                    str(min_connections) + "\tActual Connections: " + str(count))
            if max_connections is not None and count > max_connections:
                raise RuntimeError(
                    str(self.component_name) + "has an incorrect number of incoming connections.\nMaximum "
                                               "connections: " +
                    str(max_connections) + "\tActual Connections: " + str(count))

    def check_outgoing_connections(self, compartment, min_connections=None, max_connections=None):
        """
        Checks/validates the outgoing information structure/flow from this component.

        Args:
            compartment: compartment from incoming source to check

            min_connections: minimum number of incoming connections this component should receive

            max_connections: maximum number of incoming connections this component should receive
        """
        if compartment not in self._outgoing_connections.keys() and min_connections is not None:
            raise RuntimeError(
                str(self.component_name) + " has an incorrect number of outgoing connections.\nMinimum connections: " +
                str(min_connections) + "\t Actual Connections: None")

        if compartment in self._outgoing_connections.keys():
            count = self._outgoing_connections[compartment]
            if min_connections is not None and count < min_connections:
                raise RuntimeError(
                    str(self.component_name) + " has an incorrect number of outgoing connections.\nMinimum "
                                               "connections: " +
                    str(min_connections) + "\tActual Connections: " + str(count))
            if max_connections is not None and count > max_connections:
                raise RuntimeError(
                    str(self.component_name) + " has an incorrect number of outgoing connections.\nMaximum "
                                               "connections: " +
                    str(max_connections) + "\tActual Connections: " + str(count))


class Component(ABC):
    """
    Components are a foundational part of ngclearn and its component/command
    structure. In ngclearn, all stateful parts of a model take the form of
    components. The internal storage of the state within a component takes one
    of two forms, either as a compartment or as a member variable. The member
    variables are values such as hyperparameters and weights/synaptic efficacies,
    where the transfer of their individual state from component to component is
    not needed.
    Compartments, on the other hand, are where the state information, both from
    and for other components, are stored. As the components are the stateful
    pieces of the model, they also contain the methods and logic behind advancing
    their internal state (values) forward in time.

    The use of this abstract base class for components is completely optional.
    There is no part of ngclearn that strictly dictates/requires its use; however,
    starting here will provide a good foundation for development and help avoid
    errors produced from missing attributes. That being said, this is not an
    exhaustive/comprehensive base class. There are some commands such as `Evolve`
    that requires an additional method called `evolve` to be present within the
    component.
    """

    def __init__(self, name, useVerboseDict=False, **kwargs):
        """
        The only truly required parameter for any component in ngclearn is a
        name value. These names should be unique; otherwise, there will be
        undefined behavior present if multiple components in a model have the
        same name.

        Args:
            name: the name of the component

            useVerboseDict: a boolean value that controls if a more debug
                friendly dictionary is used for this component's compartments.
                This dictionary will monitor when new keys are added to the
                compartments dictionary and tell you which component key errors
                occur on. It is not recommended to have these turned on when
                training as they add additional logic that might cause a
                performance decrease. (Default: False)

            kwargs: additional keyword arguments. These are not used in the base class,
                but this is here for future use if needed.
        """
        # Component Data
        self.name = name

        self.compartments = _VerboseDict(name=self.name, **kwargs) if useVerboseDict else {}
        self.bundle_rules = {}
        self.sources = []

        # Meta Data
        self.metadata = _ComponentMetadata(name=self.name, **kwargs)

    ##Intialization Methods
    def create_outgoing_connection(self, source_compartment):
        """
        Creates a callback function to a specific compartment of the component.
        These connections are how other components will read specific parts of
        this components state at run time.

        Args:
            source_compartment: the specific compartment whose state will be
                returned by the callback function

        Returns:
            a callback function that takes no parameters and returns the state
                of the specified compartment
        """
        self.metadata.add_outgoing_connection(source_compartment)
        return lambda: self.compartments[source_compartment]

    def create_incoming_connection(self, source, target_compartment, bundle=None):
        """
        Binds callback function to a specific local compartment.

        Args:
            source: The defined callback function generally produced by
                `create_outgoing_connection`

            target_compartment: The local compartment the value should be stored in

            bundle: The bundle or input rule to be used to put the data into the
                compartment, this is overwriting the value by default, but rules
                like appending and adding are also provided in `bundle_rules.py`
        """
        self.metadata.add_incoming_connection(target_compartment)
        if bundle not in self.bundle_rules.keys():
            self.create_bundle(bundle, bundle)
        self.sources.append((source, target_compartment, bundle))

    def create_bundle(self, bundle_name, bundle_rule_name):
        """
        This tracks the all the bundle rules that the component will need while gathering information

        Args:
            bundle_name: the local name of the bundle name

            bundle_rule_name: the rule name or keyword defined in `modules.json`
        """
        if bundle_name is not None:
            rule = utils.load_attribute(bundle_rule_name)
            self.bundle_rules[bundle_name] = rule
        else:
            try:
                rule = utils.load_attribute(bundle_rule_name)
            except:
                from .bundle_rules import overwrite
                rule = overwrite
            self.bundle_rules[bundle_name] = rule

    ## Runtime Methods
    def clamp(self, compartment, value):
        """
        Sets a value of a compartment to the provided value

        Args:
            compartment: targeted compartment

            value: provided Value
        """
        self.compartments[compartment] = value

    def process_incoming(self):
        """
        Calls each callback function and yields results from all connections

        Returns:
            yields tuples of the form (value, target_compartment, bundle)
        """
        for (source, target_compartment, bundle) in self.sources:
            yield source(), target_compartment, bundle

    def pre_gather(self):
        """
        Optionally implemented method that is called before all the connections
        are processed. An example use case for this method is that if a compartment
        should be reset before multiple cables under the additive bundle rule
        are processed.
        """
        pass

    def gather(self):
        """
        The method that does all the processing of incoming value including
        calling pre_gather, and using the correct bundle rule.
        """
        self.pre_gather()
        for val, dest_comp, bundle in self.process_incoming():
            self.bundle_rules[bundle](self, val, dest_comp)

    ##Abstract Methods
    @abstractmethod
    def verify_connections(self):
        """
        An abstract method that generally uses component metadata to verify that
        there are the correct number of connections. This should error if it
        fails the verification.
        """
        pass

    @abstractmethod
    def advance_state(self, **kwargs):
        """
        An abstract method to advance the state of the component to the next one
        (a component transitions from its current state at time t to a new one
        at time t + dt)
        """
        pass

    @abstractmethod
    def reset(self, **kwargs):
        """
        An abstract method that should be implemented to models can be returned
        to their original state.
        """
        pass

    @abstractmethod
    def save(self, directory, **kwargs):
        """
        An abstract method to save component specific state to the provided
        directory

        Args:
            directory:  the directory to save the state to
        """
        pass
