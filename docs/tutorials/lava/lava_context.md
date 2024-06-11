# The Lava Context

The lava context, i.e, the `LavaContext`, serves as the core to ngc-lava as well 
as the main workhorse of all of its features. Since it is a subclass of the 
default ngc-learn context, we will only be covering the new Lava-specific 
features here.

## Building Lava Components

The Lava context generally keeps track of two sets of components -- the ngclearn 
components and the Lava components. However, due to the nature of the lava 
components themselves, they must be built once the model is fixed and cannot be 
built on-the-fly. Due to this fact, the building of the lava components must 
be triggered before they can be used. Nevertheless, there are a few ways to trigger the 
building of the Lava components. It is important to note that only the latest set 
of components can be used for methods like clamping and running. This will 
affect all dynamically compiled methods.

### Events that Trigger a Rebuild

- When a `LavaContext` is first constructed via: `with LavaContext("model") as model:` 
  leaving the context block will trigger a rebuild
- Calling `with model.updater:` will rebuild the lava components upon leaving the 
  with-block
- Calling `model.rebuild_lava()` will rebuild the lava components even if it is 
  still inside a with-block. However, by default, it will stop the with-block 
  from recompiling upon exiting as doing so would overwrite the previously built
  model components.

### Events That Will Not Trigger a Rebuild

Simply calling `with model` will not trigger a rebuild upon exiting since this is 
where additional dynamic method can be defined as well as reference sub-models 
while not triggering a complete rebuild of the Lava components each time.

## The Runtime

Inside of Lava, there is an internal runtime that is controlling the simulator 
for the loihi2. This runtime must be started in order to act upon Lava components, 
such as clamping values to their compartments as well as probing information
about the model. To help simplify this, the `LavaContext` comes with a built-in 
runtime manager. To gain access to the ngc-lava runtime manager, first call 
`model.set_up_runtime()`. Note that the `set_up_runtime` method takes two 
arguments. The first is the root Lava component name to be used to start the 
runtime -- this is how Lava knows what component it will need to simulate. The
second argument is the "rest" image -- the "rest" image is used to allow the 
dynamical system that is your model to return to its reset state while 
receiving no input (this is akin to allowing a biophysical neural system to relax 
to its resting potential state). This can be left as `None` and doing so will 
skip this functionality. Note that this method <b>does not</b> actually start 
the runtime, it just configures everything. It is important to observe that a 
clamp method fitting the signature `clamp(x) -> None` needs to be defined in 
order to use certain runtime methods as defined below.

### Runtime Methods

- `with model.runtime`: The lava runtime will exist for the duration of this 
   with block.
- `model.start_runtime()`: This starts the runtime without the management of 
   automatically stopping it later.
- `model.pause()`: Pauses the runtime, allowing for values to be read and set.
- `model.stop()`: Stops the runtime, runtimes can not be restarted once they are 
   stopped.
- `model.run(t)`: Runs the runtime, for `t` time steps. Will automatically pause 
   upon completion.
- `model.view(x, t)`: First calls `model.clamp(x)` and then runs the runtime for 
  `t` steps. Will automatically pause upon completion.
- `model.rest(t)`: First calls `model.clamp(rest_image)` and then runs the 
  runtime for `t` steps. Will automatically pause upon completion. If a reset 
  image was not supplied, this runtime method will not be available.

## Additional Utility Methods

### Using Lags with: `set_lag(component_name, status=True)`

In Lava, it is easy to lock your system if there is recurrence in your model.
The Lava context allows for you to temporally "lag" the values emitted by 
specific components, delaying their executation with respect to the previous 
time-step.

By default, the process pattern for a mapped Lava component is: 
`Receive values -> Process values -> Emit values`

A lagged Lava component will follow the pattern: 
`Emit values -> Receive values -> Process Values`

Example:
> There is a model that has the wiring pattern of `Z0 -> W1 -> Z1 -> W1`
> Here we can see that in order for Z1 to emit values it relies on the values
> emitted by W1. But W1 also relies on values emitted from Z1. So if we lag
> W1 it will emit last timesteps value at the start of the loop and then wait
> for the new values meaning that the value emitted by W1 will be delayed by a
> timestep, but it will no longer lock Z1 from running.

### `write_to_ngc()`

This method is designed to copy the current state of the Lava model into the 
ngc-learn model. This will do a one-to-one mapping of all of thecomponents and 
their values from Lava to ngclearn. It is important to point out that this must 
be done inside of a runtime. This is critical for saving since, in order to save 
an on-chip-trained model, it must first be written back to ngc-lava/learn 
and then to disk. By default, this is called by `model.save_to_json` if called 
inside a runtime.
