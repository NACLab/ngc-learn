from ngclib.commands import Command
from ngclib.utils import extract_args
import warnings

class Save(Command):
    """
    When training models, there is often a need to snapshot the model and save
    it to disk. The base controller in ngclib is able to save all of the
    commands, components, connections, and steps to a file in order to rebuild
    the model at a later time. However, there is a good chance that the model
    will contain components that have parts that need saving beyond the parameters
    passed in to initialize the component. ngclib solves this by providing a
    save command; this command will call a custom save method on all components
    provided to the command. This custom save method will be responsible for all
    custom values to be saved and determining where to save them inside of a
    provided directory.
    """
    def __init__(self, components=None, directory_flag=None, command_name=None,
                 **kwargs):
        """
        Required calls on Components: ['save', 'name']

        Args:
            components: a list of components to call the save function on

            directory_flag: keyword for flag for the directory to save to

            command_name: the name of the command on the controller
        """
        super().__init__(components=components, command_name=command_name,
                         required_calls=['save'])
        if directory_flag is None:
            raise RuntimeError(self.name + " requires a \'directory_flag\' to bind to for construction")
        self.directory_flag = directory_flag

    def __call__(self, *args, **kwargs):
        try:
            vals = extract_args([self.directory_flag], *args, **kwargs)
        except RuntimeError:
            warnings.warn(self.name + ", " + str(self.directory_flag) + " is missing from keyword arguments and no "
                                                                "positional arguments were provided", stacklevel=6)
            return

        if vals[self.directory_flag]:
            for component in self.components:
                self.components[component].save(vals[self.directory_flag])
