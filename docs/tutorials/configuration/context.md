# Contexts

Contexts, in NGC-Sim-Lib, are the top-level containers that hold everything used to
define a model / dynamical system. On their own, contexts have no runtime logic; 
they rely on their internal processes and components to build a complete, working model.

## Defining a Context

To define a context (`ngcsimlib.Context`), NGC-Sim-Lib leverages the `with` block; this 
means that to create a new context, simply start with the statement 
`with Context("myContext") as ctx:` and a new context will be created. 
(<i>Important Note</i>: names are unique; if a context is created with the same name, 
they will be the same context and, thus, there might be conflicts). 
A defined context does not do anything on its own.

## Adding Components

To add components to a context, simply initialize components while inside
the `with` block of the context. Any component defined while inside this block
will automatically be added and tacked-on to the context object.

## Wiring Components

Inside of a model / dynamical system, components will need to pass data to one 
another; this is configured within the context. To connect the compartments of 
two components, follow the pattern: `mySource.output >> myDestination.input`, 
where `output` and `input` are compartments inside their respective components. 
This format will ensure that, when processes are being run, the value will 
flow properly from component to component.

### Operators

There is a special type of wire called an operator; this performs a simple
operation on the compartment values as the data flows from one component to
another. Generally, these are use for simple mathematical operations, such as
negation `Negate(mySource.output) >> myDestination.input` or the summation of
multiple compartments into
one `Summation(mySource1.output, mySource2.output, ...) >> myDestination.input`.
Note that operators can be chained, so it would be possible to negate one or 
more of the inputs that flow into the summation.

## Adding Processes

To add processes to a context, simply initialize the process and add all of its
steps while inside the `with`-block of the process.

## Exiting the `with` block

When the context exits the `with`-block, it will re-compile the entire model. 
Behind the scenes, this is calling `recompile` on the context 
itself; it is possible to manually trigger the recompile step, but doing so can 
break certain connections (between components/compartments), so use this 
functionality sparingly.

## Saving and Loading

The context's one unique job is the handling of the "saving" (serialization) and 
"loading" (de-serialization) of models to disk. By default, calling 
`save_to_json(...)` will create the correct file structure as well as the core files 
needed and load the context in the future. To load / de-serialize a model, 
calling `Context.load(...)` will load the context in from a directory; something
important to note is that loading in a context entails effectively 
recreating the components with their initial values using their arguments as well as
keywords arguments (excluding those that cannot be serialized). This means that, 
if you have a trained model, ensure that your components have a save method 
defined that will handle the saving and loading of all values within their compartments.  

