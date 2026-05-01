# Compiling

The term "compiling" for NGC-Sim-Lib refers to automatic step that happens
inside of a context that produces a transformed method for all of its
components. This step is the most complicated part of the library and, in
general, does not need to be touched or interacted with. Nevertheless, this
section will cover most of the steps that the NGC-Sim-Lib compilation process
does at a high level. This section contains advanced technical/developer-level
information: there is an expectation that the reader has an understanding of
Python abstract syntax trees (ASTs), Python namespaces, and how to
dynamically compile Python code and execute it.

## The Decorator

In NGC-Sim-Lib, there is a decorator marked as `@compilable` which is used to
add a flag to methods that the user wants to compile. On its own, this will not
do anything; however, this decorator lets the parser distinguish between methods
that should be compiled and methods that should be ignored.

## The Step-by-Step NGC-Sim-Lib Parsing Process

The process starts by telling the parser to compile a specific object.

### Step 1: Compile Children

The first step to compile any object is to make sure that all of the 
"compilable" objects of the top level object are compiled. As a
result, NGC-Sim-Lib will loop through all of the whole object and will compile
each part that it finds that is flagged as compilable (via the decorators
mentioned above) and is, furthermore, an instance of a class.

### Step 2: Extract Methods to Compile

While the parser is looping through all of the parts of the top-level object, it
is also extracting the methods on/embedded to the object that are flagged as
compilable (with the decorator above). NGC-Sim-Lib stores them for later;
however, this lets the parser only loop over the object once.

### Step 3: Parse Each Method

As each method is its own entry-point into the transformer, this step will run 
for each method in the top-level object.

### Step 3a: Set up a Transformer

This step sets up a `ContextTransformer`, which further makes use of a
`ast.NodeTransformer`, and will convert methods from class methods (with the use
of `self`), as well as other methods that need to be removed / ignored, into
their more context-friendly counterparts.

### Step 3b: Transform the Function

There are quite a few pieces of common Python that need to be transformed. This
step happens with the overall goal of replacing all object-focused parts with a
more global view. This means that a compartment's `.get` and `.set` calls are
replaced with direct setting and getting from the global state, based on the
compartment's target. This also means that all temporally constant values --
such as `batch_size` -- are moved into the globals space for that specific file
and ultimately replaced with the naming convention of `object_path_constant`.
One more key step that is performed is to ensure that there is no branching in
the code. Specifically, if there is a branch, i.e., an if-statement, NGC-Sim-Lib
will evaluate it and only keep the branch it will traverse down. This means that
there cannot be any branch logic based on inputs or computed values (this is a
common restriction for just-in-time compiling).

### Step 3c: Parse Sub-Methods

Since it is possible to have other class methods that are not marked as
entry-points for compilation but still need to be compiled, as step 3b happens,
NGC-Sim-Lib tracks all of the sub-methods required. Notably, this step goes
through and repeats steps 3a and 3b for each of the (sub-)methods with a naming
convention similar to the temporally constant values for each method.

### Step 3d: Compile the Abstract Syntax Tree (AST)

Once we have all of the namespace and globals needed to execute the
properly-transformed method, the method is compiled with Python and finally
executed.

### Step 3e: Binding

The final step per method is to bind each to their original method; this
replaces each method with an object which, when called, will act like the
normal, uncompiled version but has the addition of the `.compiled` attribute.
This attribute contains all of the compiled information to be used later (for
model / system simulation). This crucially allows for the end user to
call `myComponent.myMethod.compiled()` and have it run. The exact type for
a `compiled` value can be found
in `ngcsimlib._src.parser.utils:CompiledMethod`.

### Step 4: Finishing Up / Final Processing

Some objects, such as the processes, entail additional steps to modify
themselves or their compiled methods in order to align themselves with needed
functionality. However, this operation/functionality is found within each
class's expanded `compile` method and should be referred to by looking at those
methods specifically.   
 
