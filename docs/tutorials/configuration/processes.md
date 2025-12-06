# Processes

Processes in NGC-Sim-Lib offer a central way of defining a specific transition to be 
taken within a given model (this effectively sets up the behavior of the state-machine 
that defines the desired dynamical system one wants to simulate). In effect, processes 
take in as many compilable methods as possible across any number of
components; they work to produce a single top-level method and a varying number of
sub-methods needed to execute the entire chain of compilable methods in one (single) step.
This is ultimately done to interface nicely with just-in-time (JIT) compilers, such as 
the one inherent to JAX, and to minimize the amount of read and write calls done across 
a chain of methods.

## Building the (Command) Chain

Building the chain that a process will use is done through an iterative process. Once
the process object is created, steps are added using either `.then()` or `>>`.
As an example: 

```
myProcess.then(myCompA.forward).then(myCompB.forward).then(myCompA.evolve).then(myCompB.evolve)
```

or

```
myProcess >> myCompA.forward >> myCompB.forward >> myCompA.evolve >> myCompB.evolve
```

In both cases, this process will chain the four methods together into a single
step, only updating the final state after all steps are complete.

## Types of Processes

There are two types of processes: the above example would be with what is
referred to as a `MethodProcess` -- these are used to chain together any
compilable methods from any number of different components. The other second 
type of process, called a `JointProcess` in NGC-Sim-Lib, is used to chain 
together entire processes. 
JointProcesses are especially useful if there are multiple method processes that
need to be called but different orders of the processes are needed at different
times. These allow for the specification of complex events / behaviors in a 
dynamical system that one will simulate. 

## Extra Elements

There are a few extra methods that come standard with each process type which can
be useful for both regular operation as well as debugging.

### Viewing the Compiled Method

Behind the scenes, a process is transforming and compiling down all of the steps
used to build it; this means that the exact code it is running to do its
set of calculations will ultimately not be what the user wrote. To allow for 
the end user to view and make sure that the two pieces of code -- theirs and 
the compiled version -- are equivalent (and yielding expected behavior), every 
process has a `view_compiled_method()` call which can be used after the (final) model 
is compiled. This call will return the code (block) that it will be running as a 
string. There will be some stark differences between the produced/generated code and
the code in the (Python) components used to build the steps. Please refer to the
compiling page for a more in-depth guide to comparing the outputs between these 
two stages of code.

### Needed Keywords

Since some methods will require external values such as `t` (for time) or `dt` 
(for integration time / the temporal delta) for a given execution, a process 
will also track all the keyword arguments that are needed to run their compiled 
process. To view which keywords a given process is expecting, one may use the 
command: `get_keywords()`. 
This is mostly used for debugging and/or as a sanity check.

### Packing Keywords

To add onto the needed keywords, the process also provides an interface to
produce the keywords needed to run in the form of two methods. The first method
is `pack_keywords(...)`; this method packs together a single row of values that 
are needed to run a single execution (step) of the process. The arguments are
the `row_seed`, which is a seed that is to be passed to all of the keyword 
generators (only needed if generators are being used). 
The second set of arguments are keyword arguments that are either constant, 
such as `dt=0.1`, or generators, such as `lambda row_seed: 0.1 * row_seed`. 
The second method for generating the keywords for a process is with `pack_rows(...)`. 
This method will create many sets of keywords that are needed to run multiple 
iterations of the process. Note that the arguments are slightly different: first, 
it now utilizes a `length` argument to indicate the number of rows being produced and, 
second, it features a `seed_generator` that is used to generate the seed of each row 
(for instance, to have only even seed values: `seed_generator = lambda x: 2 * x`); if 
the generator is `None`, then `seed_generator = lamda x: x` is used. 
After this, the same keyword arguments to define the needed parameters are used as in `pack_keywords`. 

