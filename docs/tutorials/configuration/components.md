# Components

Living one step above compartments in the NGC-Learn dynamical systems hierachy rests the component. 
A component (`ngcsimlib.Component`) holds a collection of both temporally constant values as well as dynamic (time-evolving) 
compartments. In addition, they are the core place where logic governing the dynamics of a system are 
defined. Generally, components serve as the building blocks that are to be reused multiple times 
when constructing a complete model of a dynmical system.

## Temporally Constant versus Dynamic Compartments

One important distinction that needs to be highlighted within a component is the 
difference between a temporally constant value and a dynamic (time-varying) compartment.
Compartments themselves house values that change over time and, generally, they will have the 
type `ngcsimlib.Compartment`; note that compartments are to be used to track the internal values 
of a component. These internal values can be ones such inputs, decaying values, counters, etc. 
The second kind of values found within a component are known as temporally constant values; these 
are values (e.g., hyper-parameters, structural parameters, etc.) that will remain fixed 
within constructed model dynamical system. These types of values tend to include common configuration 
and meta-parameter settings, such as matrix shapes and coefficients.

## Defining Compilable Methods

Inside of a component, it is expected that there will be methods defined that govern the
temporal dynamics of the system component. These compilable methods are decorated 
with `@compilable` and are defined like any other regular (Python) method. Within a compilable
method, there will be access to `self`, which means that, to reference a compartment's
value, one must write out such a call as: `self.myCompartment.get()`. The only requirement is 
that any method that is decorated <b>cannot</b> have a return value; values should be stored 
inside their respective compartments (by making an appeal to their respective set routine, i.e., 
`self.myCompartment.set(value)`). In an external (compilation) step, outside of the developer's 
definition of a component, an NGC-Sim-Lib transformer will change/convert all of these (decorated) 
methods into ones that function with the rest of the NGC-Sim-Lib back-end.
