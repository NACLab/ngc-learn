# Understanding Commands

## Overview
Commands are one of the central pillars of
<a href="https://github.com/NACLab/ngc-sim-lib">ngcsimlib</a>, the dependency
library that drives ngc-learn's simulation backend.
In general, commands provide the instructions and logic for what each component
should be doing at any given time. In addition, they are the normal way that an
outside user would interact with ngc-learn models. Commands live inside a model's  
controller and are generally made with the `add_command` method.

## Abstract Command
Contained within ngcsimlib is an abstract class for every command included in
ngcsimlib. It is strongly recommended that custom commands are built using this
base class (but there is nothing enforcing this inside of ngcsimlib).

At its base the abstract command forces two things: firstly, the constructor
for the base class requires a list of components, and a list of attributes that
each component should have. Secondly, all commands must implement their
`__call__` command, taking in only `*args` and `**kwargs`.

## Constructing Commands
It is common that commands will need to have values passed into them to control
their internal behavior, such as a value to clamp, or a flag for freezing
synaptic weight values.
To do this, we introduce the notion of binding keywords to commands.
Specifically, commands will take strings in during their construction and then
look for those strings when called inside the list of keyword arguments in order
to get their arguments.

## Calling Commands
When commands are called, they will take in only `*args` and `**kwargs`.
While custom commands can break this by adding in additional arguments
without any problem, it is not recommended to do this as multiple instances
of a command with different parameters will then use the same keyword for their
call.

## Creating Custom Commands
It is recommended that all custom commands inherit from the base class
provided within ngcsimlib. This provides a good starting point for designing a
component that will seamlessly interact with ngcsimlib's internal simulation mechanics.
These mechanics, which characterize the core operation of a simulation controller,
entail that, for each command supplied to a controller, a command will call the
same function with the same parameters on each component provided
to that very command. It is also expected that there is error handling within the
constructor to catch as many runtime errors as possible. Note that base
command class provides a list to check required calls such as `reset` or `evolve`.

It is important to note that, if commands are going to be constructed via a
controller, they should have keyword arguments with default values that
error out on bad input instead of positional arguments.

## Example Command (reset)

Below, we present the key bits of source code that characterize a reset command
-- a very commonly used, built-in command for models designed in ngc-learn -- and
its internal operation:

```python
from ngcsimlib.commands import Command
from ngcsimlib.utils import extract_args
from ngcsimlib.logger import warn, error

class Reset(Command):
    def __init__(self, components=None, reset_name=None, command_name=None,
                 **kwargs):
        super().__init__(components=components, command_name=command_name,
                         required_calls=['reset'])
        if reset_name is None:
            error(self.name, "requires a \'reset_name\' to bind to for construction")
        self.reset_name = reset_name

    def __call__(self, *args, **kwargs):
        try:
            vals = extract_args([self.reset_name], *args, **kwargs)
        except RuntimeError:
            warn(self.name, ",", self.reset_name,
                 "is missing from keyword arguments and no positional arguments were provided")
            return

        if vals[self.reset_name]:
            for component in self.components:
                self.components[component].reset()
```

## Custom Command Template

Here, we show the generic command template which shows how one would go about
designing the key operational bits that make up a useful command.

```python
from ngcsimlib.commands.command import Command
from ngcsimlib.utils import extract_args
from ngcsimlib.logger import error


class CustomCommand(Command):
    def __init__(self, components=None, BINDING_VALUE=None, ADDITIONAL_INPUT=None, command_name=None,
                 **kwargs):
        super().__init__(components=components, command_name=None, required_calls=['CUSTOM_CALL'])
        # Make sure additional input is passed in
        if ADDITIONAL_INPUT is None:
            error(self.name, "requires a \'ADDITIONAL_INPUT\' for construction")

        # Make sure command is bound to a value
        if BINDING_VALUE is None:
            error(self.name, "requires a \'BINDING_VALUE\' to bind to for construction")

        self.BOUND_VALUE = BINDING_VALUE
        self.ADDITION_VALUE = ADDITIONAL_INPUT

    def __call__(self, *args, **kwargs):
        # Extract the bound value from the arguments
        try:
            vals = extract_args([self.BOUND_VALUE], *args, **kwargs)
        except RuntimeError:
            error(self.name, ",", str(self.BOUND_VALUE), "is missing from keyword arguments or a positional "
                                                                  "arguments can be provided")

        #Use extracted value to call a method on each component
        for component in self.components:
            self.components[component].CUSTOM_CALL(self.ADDITION_VALUE, vals[self.BOUND_VALUE])
```

## Notes
All components added to commands must have a `name` attribute and the word
`name` is automatically appended to any provided list of required attributes
to the base class constructor.

As all built-in commands use `extract_args` when called with a controller via
`myController.COMMAND(ARGUMENT)`, there is no need to use keywords as it will
use `args` if there are no keyword arguments. (Keywords will still work, however.)

When commands are constructed via a controller, they are also provided with the
keyword arguments `controller` and `command_name`. It is not recommended to
use these for any core logic (just use them for error messages), unless
it using them is absolutely essential in achieving the desired functionality.
