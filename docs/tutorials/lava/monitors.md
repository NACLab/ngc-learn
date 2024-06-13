# Monitors

While lava does have its own version of monitors so does ngclearn. It is
recommended that you use the ngclearn monitors as they have expanded
functionality and are known to interact with the lava components well. For an
overview of how monitors work reference [this](../foundations/monitors.md). The
only difference is that lava has its own monitor found
in `ngclearn.components.lava`.

## Sharp edges

- Due to the fact that a lava component of the monitor must be built it has to
  be defined inside the lavaContext.
- To view the values found in the monitor via the `view()` and `get_path()`
  methods `model.write_to_ngc()` must be called to refresh the values.