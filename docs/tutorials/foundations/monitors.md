# NGC Monitors

Ngc-monitors are a way of storing a rolling window of compartment values
automatically. Their intended purpose is not to be used inside of model but just
as an auxiliary way to viewing the internal state of the module even when
compiled. A monitor will track the last `n` values it observed in the
compartment with the oldest value being at `index=0` and the newest being at
`index=n-1`.

## Building a Monitor

Monitors are built exactly like regular components are for models. Simply import
the monitor `from ngclearn.components import Monitor`. Now inside your model
build it like a regular component.

```python
with Context("model") as model:
    M = Monitor("M", default_window_length=100)
```

## Watching compartments

There are then two ways of watching compartments, the first way looks similar
to the wiring paradigm found in connecting components together. The main
difference is that connecting compartments to the monitor does not require a
compartment, they are wired directly into the Monitor following the pattern

```python
    M << z0.s
```

This will wire the spike output of `z0` into the monitor with a view window
length of the `default_window_length`. In the event that you want a view window
of not the default viewing length you can use the `watch()` method instead.

```python
    M.watch(z0.s, customWindowLength)
```

There is no limit to the number of compartments a monitor can watch or the
length of the window it can store. However, as it is constantly shifting values,
tracking large matrices such as synapses over many timesteps may get expensive.

For the monitor to run during your `advance_state` and `reset` calls make sure
to add it to the list of components to compile. Currently, monitors do not work
with non-compiled methods (This is planned for the future).

## Extracting values

To look at the current stored window of any compartment being tracked there are
two methods. The first method requires that you have access to the compartment
it the monitor is watching. To read out the monitors values call

```python
M.view(z0.s)
```

In the event that you do not have access to the compartment all stored values
can be found via the path using

```python
M.get_store("path/to/compartment")
```

The stored windows are kept in a tree of dictionaries where each node is a part
of the path and the leaves are compartment objects holding the windows.