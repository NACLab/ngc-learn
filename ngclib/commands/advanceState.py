from ngclib.commands import Command


class AdvanceState(Command):
    """
    As the general form, all models built in ngclearn are state machines and this
    command is designed to advance the state of all of the components passed into
    the command. Prior to advancing the state of each component, this will call the
    `gather` method of that component.
    """
    def __init__(self, components=None, command_name=None, **kwargs):
        """
        Required calls on Components: ['advance_state', 'gather', 'name']

        Args:
            components: the list of components to advance the state of
            command_name: the name of the command on the controller
        """
        super().__init__(components=components, command_name=command_name,
                         required_calls=['advance_state', 'gather'])

    def __call__(self, **kwargs):
        for component in self.components:
            self.components[component].gather()
            self.components[component].advance_state(**kwargs)
