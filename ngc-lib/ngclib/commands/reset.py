from ngclib.commands import Command
from ngclib.utils import extract_args
import warnings

class Reset(Command):
    """
    In every model/system, there is a need to reset components back to some
    intial state value(s). As such, many components that maintain a state have a
    reset method implemented within them. The reset command will go through
    the list of components and trigger the reset within each of them.
    """
    def __init__(self, components=None, reset_name=None, command_name=None,
                 **kwargs):
        """
        Required calls on Components: ['reset', 'name']

        Args:
            components: a list of components to reset

            reset_name: the keyword for the flag on if the reset should happen

            command_name: the name of the command on the controller
        """
        super().__init__(components=components, command_name=command_name,
                         required_calls=['reset'])
        if reset_name is None:
            raise RuntimeError(self.name + " requires a \'reset_name\' to bind to for construction")
        self.reset_name = reset_name

    def __call__(self, *args, **kwargs):
        try:
            vals = extract_args([self.reset_name], *args, **kwargs)
        except RuntimeError:
            warnings.warn(self.name + ", " + str(self.reset_name) + " is missing from keyword arguments and no "
                                                             "positional arguments were provided", stacklevel=6)
            return

        if vals[self.reset_name]:
            for component in self.components:
                self.components[component].reset()
