# NGC Monitors

Ngc-monitors are a way of storing a rolling window of compartment values
automatically. Their intended purpose is not to be used inside of a model but 
just as an auxiliary way to view the internal state of the module even when it is 
compiled. A monitor will track the last `n` values it has observed within the
compartment with the oldest value being at `index=0` and the newest being at
`index=n-1`.

## Building a Monitor

Monitors are constructed exactly like regular components are for general models. 
Simply import the monitor `from ngclearn.components import Monitor`. Now, 
inside of your model, build it like a regular component.

```python
with Context("model") as model:
    M = Monitor("M", default_window_length=100)
```

## Watching compartments

There are then two key ways of watching compartments, the first way looks similar
to the wiring paradigm found in connecting standard ngclearn components 
together. The primary difference is that connecting compartments to the monitor 
does not require a compartment, they are wired directly into the `Monitor` 
following the pattern below:

```python
    M << z0.s
```

This will wire the spike output of `z0` into the monitor with a view window
length of the `default_window_length`. In the event that you want a view window
that is not the default viewing length, you can use the `watch()` method 
instead as in below:

```python
    M.watch(z0.s, customWindowLength)
```

There is no limit to the number of compartments that a monitor can watch or the
length of the window that it can store. However, as it is constantly shifting 
values, tracking large matrices, such as those containing synapse values 
over many timesteps, may get expensive.

For the monitor to run during your `advance_state` and `reset` calls, make sure
to add to the monitor to the list of components to compile. Currently, 
monitors do not work with non-compiled methods 
(<i>This is a planned feature for future developments of ngc-learn</i>).

## Extracting Values

To look at the currently stored window of any compartment being tracked, there 
are two methods available to you. The first method requires that you have 
access to the compartment that the monitor is watching. To read out the 
monitors values, you can call:

```python
M.view(z0.s)
```

In the event that you do not have access to the compartment, all of the stored 
values can be found via the path using the following:

```python
M.get_store("path/to/compartment")
```

The stored windows are kept in a tree of dictionaries, where each node is a part
of the path and the leaves are compartment objects holding the 
tracked value windows.