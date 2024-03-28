from ngclib.commands import Command

class Snapshot(Command):
    """
    Sometimes when running through a model, there is a need to extract a value
    from a compartment to the run loop. As such, ngclib provides the snapshot
    command. This command will extract a given attribute from all components
    provided to the command. This command returns a single value if only one
    component is given, otherwise it will return a list where each value
    corresponds to the position of the component in the components list.

    This command is regularly used for debugging/graph production code that
    exists outside of the model. It would be incorrect to clamp the output of
    this command into another component, if that is the intended goal, please
    see `connect` in the controller.
    """
    def __init__(self, components=None, attribute=None, command_name=None,
                 **kwargs):
        """
        Required calls on Components: ['name'], and the passed-in attribute

        Args:
            components: the component extract the values of

            attribute: a single attribute to return

            command_name: the name of the command on the controller
        """
        super().__init__(components=components, command_name=command_name,
                         required_calls=[attribute])
        self.attribute = attribute


    def __call__(self, *args, **kwargs):
        vals = []
        for component in self.components:
            vals.append(getattr(self.components[component], self.attribute))

        return vals if len(vals) > 1 else vals[0]
