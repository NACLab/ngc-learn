"""
Contains all the built-in bundle rules as well as the default one for an unbundled
cable. Bundle(s) usage is that it is to be added to a component's bundle rules
(Note: the overwrite rule is the default rule).


Template for bundle rules. The name of the bundle rule should be meaningful to what it does. All bundle rules will
be called with the same three inputs: component, value, and destination_compartment. These are not passed by keyword but
they will always be in the same order. The component is the component that the bundle has as a destination (target).
The value is the signal that goes into (or comes along to) the connected bundle. The destination_compartment is
the compartment that the bundle is connected to. Overall, the general usage of bundle rules is to modify the
behavior of an input to a compartment; generally, they should NOT modify any other aspect of the target
component aside from the destination compartment and, as a warning, should not try to reference compartments by
name as this will possibly result in runtime errors given that the required compartments for the bundle rule(s)
might not exist.

General bundle rule specification:

def BUNDLE_RULE_NAME(component, value, destination_compartment):
    ## Logic for processing transmitted value
    ## Syntax for referencing destination compartment -> component.compartments[destination_compartment]

"""
def overwrite(component, value, destination_compartment):
    """
    The overwrite bundle rule routine.

    Args:
        component: target component node that this bundle rule will operate on

        value: the value to insert into a compartment within the target component

        destination_compartment: compartment within component to place a value w/in
    """
    component.compartments[destination_compartment] = value


def additive(component, value, destination_compartment):
    """
    The additive/addition bundle rule routine.

    Args:
        component: target component node that this bundle rule will operate on

        value: the value to add to current state of compartment w/in target component

        destination_compartment: compartment within component to add a value to
    """
    component.compartments[destination_compartment] += value


def append(component, value, destination_compartment):
    """
    The append/aggregation bundle rule routine. This is primarily useful if
    the compartment is a list of a values/objects (an appendable list construct).

    Args:
        component: target component node that this bundle rule will operate on

        value: the value to append to current state of compartment w/in target component

        destination_compartment: compartment within component to append to
    """
    component.compartments[destination_compartment].append(value)
