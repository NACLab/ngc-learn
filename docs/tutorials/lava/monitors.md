# Monitors

While lava does have its own version of monitors, ngclearn offers an 
in-built version for convenience. It is 
recommended that you use the ngclearn monitors as they have expanded
functionality and are designed to interact with the Lava components well. For an
overview of/details on how monitors work please see 
[this](../foundations/monitors.md). The
only difference is that Lava has its own monitor found
in `ngclearn.components.lava`.

## Sharp Edges and Bits

- Due to the fact that a Lava component of the monitor must be built, it has to
  be defined inside the `LavaContext`.
- To view the values found within the monitor via the `view()` and `get_path()`
  methods, `model.write_to_ngc()` <b>must</b> be called to refresh the values.