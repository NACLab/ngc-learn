# Compartments

Within NGC-Sim-Lib, the global state serves as the backbone of any given model. 
This global state is essentially the culmination of all of the dynamic or changing parts of the model itself. Each 
value that builds this state is stored within a special "container" that helps track these changes over time -- this 
is referred to as a `Compartment`.

## Practical Information

Practically, when working with compartments, there are a few simple things to keep in mind despite the fact that most 
of NGC-Sim-Lib's primary operation is behind-the-scenes bookkeeping. The two main points to note are: 
1. Each compartment holds a value and, thus, setting a compartment with `myCompartment = newValue` will not function as 
   intended since this will overwrite the Python object, i.e., the compartment with `newValue`. Instead, it is 
   important to make use of the `.set()` method to update the value stored inside a compartment so 
   `myCompartment = newValue` becomes `myCompartment.set(newValue)`.
2. In order to retrieve a value from a compartment, use `myCompartment.get()`. These methods of getting and setting 
   data inside a compartment are important to use when both working with and designing a multi-compartment component 
   (i.e., `Component`).

## Technical Information

The follow sections are devoted to explication of more technical information regarding how a compartment functions 
with in the broader scope of NGC-Sim-Lib and, furthermore, to explain how to leverage this information.

### How Data is Stored (Within a Model Context)

The data stored inside of a compartment is not actually physically stored within a compartment. Instead, it is stored 
inside of the global state and each compartment effectively holds the path or `key` to the right spot in the global 
state, allowing it to pull out a specific piece of information. As such, it is technically possible to manipulate the 
value of a compartment without actually touching the compartment object itself within any given component. By default, 
compartments have in-built safeguards in order to prevent this from happening accidentally; however, directly addressing 
the compartment within the global state directly has no such safeguards.

### What is "Targeting"?

As discussed in the model building section, there is notion of "wiring" together different compartments of different 
components -- this is at the core of NGC-Learn's and NGC-Sim-Lib's "nodes-and-cables system". These wires are created 
through the concept of "targeting,", which is, in essence, just the updating of the path stored within a compartment 
using the path of a different compartment. This means that, if the targeted compartment goes to retrieve the value 
stored within it, it will actually retrieve the value of a different compartment (as dictated by the target). When a 
compartment is in this state -- where it is targeting another compartment -- it is set to read-only, which only means that 
it cannot modify a different compartment.


