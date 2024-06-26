import json

from ngclearn import Component, Compartment
from ngclearn import numpy as np
from ngcsimlib.utils import add_component_resolver, add_resolver_meta, get_current_path
from ngcsimlib.logger import warn, critical


class Base_Monitor(Component):
    """
    An abstract base for monitors for both ngclearn and ngclava. Compartments
    wired directly into this component will have their value tracked during
    `advance_state` loops automatically.

    Note the monitor only works for compiled methods currently


    Using default window length:
        myMonitor << myComponent.myCompartment

    Using custom window length:
        myMonitor.watch(myComponent.myCompartment, customWindowLength)

    To get values out of the monitor either path to the stored value directly, or pass in a compartment directly. All
    paths are the same as their local path variable.

    Using a compartment:
        myMonitor.view(myComponent.myCompartment)

    Using a path:
        myMonitor.get_store(myComponent.myCompartment.path).value

    There can only be one monitor in existence at a time due to the way it interacts with resolvers and the compilers
    for ngclearn.

    Args:
        name: The name of the component.

        default_window_length: The default window length.
    """

    _singleton = None  # Only one Monitor

    @staticmethod
    def build_advance(compartments):
        """
        A method to build the method to advance the stored values.

        Args:
            compartments: A list of compartments to store values

        Returns: The method to advance the stored values.

        """
        critical(
            "build_advance() is not defined on this monitor, use either the monitor found in ngclearn.components or "
            "ngclearn.components.lava (If using lava)")


    @staticmethod
    def build_reset(compartments):
        """
        A method to build the method to reset the stored values.
        Args:
            compartments: A list of compartments to reset

        Returns: The method to reset the stored values.
        """
        @staticmethod
        def _reset(**kwargs):
            return_vals = []
            for comp in compartments:
                current_store = kwargs[comp + "*store"]
                return_vals.append(np.zeros(current_store.shape))
            return return_vals if len(compartments) > 1 else return_vals[0]

        return _reset

    def __init__(self, name, default_window_length=100, **kwargs):
        if Base_Monitor._singleton is not None:
            critical("Only one monitor can be built")
        else:
            Base_Monitor._singleton = True
        super().__init__(name, **kwargs)
        self.store = {}
        self.compartments = []
        self._sources = []
        self.default_window_length = default_window_length

    def __lshift__(self, other):
        if isinstance(other, Compartment):
            self.watch(other, self.default_window_length)
        else:
            warn("Only Compartments can be monitored not", type(other))

    def watch(self, compartment, window_length):
        """
        Sets the monitor to watch a specific compartment, for a specified window length.

        Args:
            compartment: the compartment object to monitor

            window_length: the window length
        """
        cs, end = self._add_path(compartment.path)

        shape = compartment.value.shape
        new_comp = Compartment(np.zeros(shape))
        new_comp_store = Compartment(np.zeros((window_length, *shape)))

        comp_key = "*".join(compartment.path.split("/"))
        store_comp_key = comp_key + "*store"

        new_comp._setup(self, comp_key)
        new_comp_store._setup(self, store_comp_key)

        new_comp << compartment

        cs[end] = new_comp_store
        setattr(self, comp_key, new_comp)
        setattr(self, store_comp_key, new_comp_store)
        self.compartments.append(new_comp.path)
        self._sources.append(compartment)
        self._update_resolver()

    def halt(self, compartment):
        """
        Stops the monitor from watching a specific compartment. It is important
        to note that it does not stop previously compiled methods. It does not
        remove it from the stored values, so it can still be viewed.
        Args:
            compartment: The compartment object to stop watching
        """
        if compartment not in self._sources:
            return

        comp_key = "*".join(compartment.path.split("/"))
        store_comp_key = comp_key + "*store"

        self.compartments.remove(getattr(self, comp_key).path)
        self._sources.remove(compartment)

        delattr(self, comp_key)
        delattr(self, store_comp_key)
        self._update_resolver()

    def halt_all(self):
        """
        Stops the monitor from watching all compartments.
        """
        for compartment in self._sources:
            self.halt(compartment)
            
    def _update_resolver(self):
        output_compartments = []
        compartments = []
        for comp in self.compartments:
            output_compartments.append(comp.split("/")[-1] + "*store")
            compartments.append(comp.split("/")[-1])

        args = []
        parameters = []

        add_component_resolver(self.__class__.__name__, "advance_state",
                               (self.build_advance(compartments), output_compartments))
        add_resolver_meta(self.__class__.__name__, "advance_state",
                          (args, parameters, compartments + [o for o in output_compartments], False))

        add_component_resolver(self.__class__.__name__, "reset", (self.build_reset(compartments), output_compartments))
        add_resolver_meta(self.__class__.__name__, "reset",
                          (args, parameters, [o for o in output_compartments], False))

    def _add_path(self, path):
        _path = path.split("/")[1:]
        end = _path.pop(-1)

        current_store = self.store
        for p in _path:
            if p not in current_store.keys():
                current_store[p] = {}
            current_store = current_store[p]

        return current_store, end

    def view(self, compartment):
        """
        Gets the value associated with the specified compartment

        Args:
            compartment: The compartment to extract the stored value of

        Returns: The stored value, None if not monitoring that compartment

        """
        _path = compartment.path.split("/")[1:]
        store = self.get_store(_path)
        return store.value if store is not None else store

    def get_store(self, path):
        current_store = self.store
        for p in path:
            if p not in current_store.keys():
                return None
            current_store = current_store[p]
        return current_store

    def save(self, directory, **kwargs):
        file_name = directory + "/" + self.name + ".json"
        _dict = {"sources": {}, "stores": {}}
        for key in self.compartments:
            n = key.split("/")[-1]
            _dict["sources"][key] = self.__dict__[n].value.shape
            _dict["stores"][key + "*store"] = self.__dict__[n + "*store"].value.shape

        with open(file_name, "w") as f:
            json.dump(_dict, f)

    def load(self, directory, **kwargs):
        file_name = directory + "/" + self.name + ".json"
        with open(file_name, "r") as f:
            vals = json.load(f)

            for comp_path, shape in vals["stores"].items():

                compartment_path = comp_path.split("/")[-1]
                new_path = get_current_path() + "/" + "/".join(compartment_path.split("*")[-3:-1])

                cs, end = self._add_path(new_path)

                new_comp = Compartment(np.zeros(shape))
                new_comp._setup(self, compartment_path)

                cs[end] = new_comp
                setattr(self, compartment_path, new_comp)



            for comp_path, shape in vals['sources'].items():
                compartment_path = comp_path.split("/")[-1]
                new_comp = Compartment(np.zeros(shape))
                new_comp._setup(self, compartment_path)

                setattr(self, compartment_path, new_comp)
                self.compartments.append(new_comp.path)

            self._update_resolver()
