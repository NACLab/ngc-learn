from ngclib.commands.command import Command
from ngclib.utils import extract_args


class Clamp(Command):
    """
    All components in ngclearn have compartments where they store information
    pertaining to their internal state that can be read into or out of by
    commands. The Clamp command is the primary way to manually set the value of
    a compartment on a set of components. The Clamp command requires a
    compartment that a value can be passed into and be clamped to,
    as well as a `clamp_name` used to locate the value when called.
    """

    def __init__(self, components=None, compartment=None, clamp_name=None,
                 command_name=None, **kwargs):
        """
        Required calls on Components: ['clamp', 'name']

        Args:
            components: a list of components to call clamp on

            compartment: the compartment being clamped to

            clamp_name: a keyword to bind the input for this command do

            command_name: the name of the command on the controller

        """
        super().__init__(components=components, command_name=command_name,
                         required_calls=['clamp'])
        if compartment is None:
            raise RuntimeError(
                self.name + " requires a \'compartment\' to clamp to for construction")
        if clamp_name is None:
            raise RuntimeError(
                self.name + " requires a \'clamp_name\' to bind to for construction")

        self.clamp_name = clamp_name
        self.compartment = compartment

        for component in self.components:
            if self.compartment not in self.components[component].compartments.keys():
                raise RuntimeError(self.name + " is attempting to "
                                               "initialize clamp to "
                                               "non-existent compartment.")

    def __call__(self, *args, **kwargs):
        try:
            vals = extract_args([self.clamp_name], *args, **kwargs)
        except RuntimeError:
            raise RuntimeError(self.name + ", " + str(
                self.clamp_name) + " is missing from keyword arguments or a positional "
                                   "arguments can be provided")

        for component in self.components:
            self.components[component].clamp(self.compartment, vals[self.clamp_name])

