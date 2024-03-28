from abc import ABC, abstractmethod
from ngclib.utils import check_attributes


class Command(ABC):
    """
    The base class for all commands found in ngclib. At its core, a command is
    essentially a method to be called by the controller that affects the
    components in a simulated complex system / model in some way. When a command
    is made, a preprocessing step is run to verify that all of the needed
    attributes are present on each component. Note that this step does not
    ensure types or values, just that they do or do not exist.
    """
    def __init__(self, components=None, command_name=None, required_calls=None):
        """
        Required calls on Components: ['name']

        Args:
            components: a list of components to run the command on

            required_calls: a list of required attributes for all components

            command_name: the name of the command on the controller
        """
        self.name = str(command_name)
        self.components = {}
        required_calls = ['name'] if required_calls is None else required_calls + ['name']
        for comp in components:
            if check_attributes(comp, required_calls, fatal=True):
                self.components[comp.name] = comp

    @abstractmethod
    def __call__(self, *args, **kwargs):
        pass
