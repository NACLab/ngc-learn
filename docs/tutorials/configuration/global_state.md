# The Global State

Since NGC-Sim-Lib is a simulation library focused on temporal models and dynamical 
systems, or models that change over time there, it is foundational that all models 
(and their respective elements) have some concept of a "state". These states
might be comprised of a single value that changes/evolves or of a complex set of values
that, when combined all together, make up the full dynamical system that underwrites the 
final model. In both cases, these sets of values are stored in what is known as the 
<i>global state</i>.

## Interacting with the Global State

Since the global state will contain a large amount of information describing a given
model, there will be a need to facilitate interaction with and modification of the values 
contained within the global state. In most use-cases, this is not done directly. The 
most common way to interact with the global state is through the use of the state-manager. 
The state-manager exists to provide a set of helper methods for interacting with the
global state itself. Note that, although the manager is there to assist you, it will not stop
you from changing the state (or "breaking" the state). When changing the state -- beyond 
setting it through the specificaiton of <i>processes</i> -- be careful to not add or remove
anything that is needed for your actual model.

### Adding New Fields to the Global State

If you are new to using NGC-Sim-Lib and looking for a way to add values to the
global state directly and explicitly, stop for a moment and reconsider. Unless 
you know exactly what you are doing (i.e., doing core development), it is strongly 
advised to not manually add values to the global state; instead, work through the 
mechanisms afforded by compartments and/or components, as these are built to afford you the 
most common ways for adding fields to the global state itself. The dynamical systems 
semantics inherent to compartments and components is meant to ensure carefully-constrained 
design and simulation of flexible models. 

If you actually intend to manually and directly add values to the global state itself, it 
is done through the use of the `add_key` method. This will create the appropriate key in
the global state for the given path and name; furthermore, its value can be retrieved
with `from_key` calls. This value, however, is not linked to a compartment and, therefore, 
will be hard to get working properly in the compiled methods without some specific references. 
<i>Please take extra care when working directly and explicitly with the global state.</i>

### Getting the Current State

To get the current state, simply call `global_state_manager.state`; this will give
you a (shallow) copy of the current state, which means that any modifications made to it will 
not be reflected in the global state.

### Updating the Global State

To manually update the global state after modifying a local copy; please write an overriding 
call command: `global_state_manager.state = new_state`. This will update the state with the 
`.update` call to its underlying dictionaries, which means that a partial state will still update correctly.

