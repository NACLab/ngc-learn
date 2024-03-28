from ngclib.commands import Command
from ngclib.utils import extract_args

class Evolve(Command):
    """
    In many models, there is a need to have either a backward pass or a separate
    method to update some particular internal state (value), e.g., a "learning"
    or evolutionary change that is not related to central compartment states.
    In general, this can be mapped to a call of `evolve`. Like with
    `advanceState`, this will call the gather method prior to calling the
    `evolve` function of every component.

    """
    def __init__(self, components=None, frozen_flag=None, command_name=None,
                 **kwargs):
        """
        Required calls on Components: ['evolve', 'gather', 'name']

        Args:
            components: the list of components to evolve

            frozen_flag: the keyword for the flag to freeze this evolve step

            command_name: the name of the command on the controller
        """
        super().__init__(components=components, command_name=command_name,
                         required_calls=['evolve'])

        self.frozen_flag = frozen_flag

    def __call__(self, *args, **kwargs):
        vals = {}
        try:
            vals = extract_args([self.frozen_flag], *args, **kwargs)
        except RuntimeError:
            vals[self.frozen_flag] = False

        if not vals[self.frozen_flag]:
            for component in self.components:
                self.components[component].gather()
                self.components[component].evolve(**kwargs)
