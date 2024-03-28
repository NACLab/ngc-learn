from ngclib.commands import Command
from ngclib.utils import extract_args
import warnings

class Track(Command):
    """
    When running a model or complex system, there is often a need to track a
    compartment value over time, usually for visualization. To do this, ngclib
    provides a track utility command. This command stores the values of a
    compartment from a set of components into a provided object. This provided
    object is expected to have an `.append` method implemented and each element
    appended to this object will be the values of the compartment for each
    provided component, in the same order that they were provided in.
    """
    def __init__(self, components=None, compartment=None, tracker=None,
                 command_name=None, **kwargs):
        """
        Required calls on Components: ['name']

        Args:
            components: a list of components to track values from

            compartment: the compartment to extract information from

            tracker: the keyword for which the tracking object will be passed in by

            command_name: the name of the command on the controller
        """
        super().__init__(components=components, command_name=command_name)
        if compartment is None:
            raise RuntimeError(self.name + " requires a \'compartment\' to clamp to for construction")
        if tracker is None:
            raise RuntimeError(self.name + " requires a \'tracker\' to bind to for construction")

        self.tracker = tracker
        self.compartment = compartment


    def __call__(self, *args, **kwargs):
        try:
            vals = extract_args([self.tracker], *args, **kwargs)
        except RuntimeError:
            warnings.warn(self.name + ", " + str(self.tracker) + " is missing from keyword arguments and no "
                                                                "positional arguments were provided", stacklevel=6)
            return

        v = []
        for component in self.components:
            v.append(self.components[component].compartments[self.compartment])
        vals[self.tracker].append(v)
