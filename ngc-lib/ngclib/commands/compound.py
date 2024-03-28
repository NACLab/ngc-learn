from ngclib.commands import Command
from ngclib.utils import check_attributes
import warnings

class Compound(Command):
    """
    There is sometimes a need by controllers to be able to call a set of commands
    in series without writing custom command(s) for each combination. The
    compound node/command is used to fill this need. A compound command set is
    very similar to the cycle found inside/within a controller / simulation object.
    """
    def __init__(self, components=None, command_name=None, command_list=None,
                 controller=None, **kwargs):
        """
        Constructs a compound command construct.
        
        Args:
            components: a list of components, this will go unused by default

            command_name: the name of the compound command

            command_list: a list of all commands to be called

            controller: the controller that will be calling these commands
        """
        super().__init__(components=components, command_name=command_name)
        if controller is None:
            raise RuntimeError("The controller is needed to build a compound command (This should be passed in by default)")
        if command_list is None or len(command_list) == 0:
            warnings.warn("The command list for command " + self.name + " is None or empty")

        self.command_list = command_list
        self.controller = controller

        check_attributes(self.controller, self.command_list, fatal=True)

    def __call__(self, *args, **kwargs):
        for command in self.command_list:
            self.controller.runCommand(command, *args, **kwargs)
