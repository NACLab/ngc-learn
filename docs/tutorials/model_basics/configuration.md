# Lesson 1: Configuring NGC-Sim-Lib

## Basics

There are various global configurations that can be made to NGC-Sim-Lib, the
systems simulation backend for NGC-Learn. The primary built in use for a 
configuration file is to modify the built-in logger. Generally to control the
configuration running any script with the flag 
`--config="path/to/your/config.json`.

The `config.json` file contains one large json object with sections set up for
different parts of the configuration, broke up into sub-objects. There is no
limit to the size or the number of these objects, meaning that the user is free
to define and use them as they so choose.

### Logging

The logging configuration mechanism sets up and controls the instance of the
python logger built into ngcsimlib. This mechanism (or JSON section) has three
values found within it. Specifically, `logging_level`, `logging_file`,
and `hide_console`. The logging levels are the same ones built into the python
logger and the value words used are either the standard Python string
representation of the level or the numeric equivalent. The `logging file`, if
defined, is a file that the logger will append all logging messages to for a
more permanent history of all messages. Finally, `hide console`, if set to true,
will hide all logging output to the console.

> Default Config
> ```json
> {
>   "logging": {
>     "logging_level": "ERROR",
>     "hide_console": false
>   }
> }
> ```

> Example Config (Write everything to file)
> ```json
> {
>   "logging": {
>     "logging_file": "path/to/log/file.txt",
>     "logging_level": "INFO",
>     "hide_console": true
>   }
> }
> ```

## Using a Configuration

To use a configuration, there are a few options. The first option is to simply
use the configuration as a python dictionary. This is done by importing
the `get_config` method from `ngclearn` and providing the name of
the configuration section to the method.

> Example get_config
>```python
>from ngclearn import get_config
> 
>loggerConfig = get_config("logger")
>level = loggerConfig['logging_level']
>```

The other way you can access a configuration is through a provided namespace.
This makes use of python's `SimpleNamespace` to map all the dictionary's key
values to properties of an object to be used. One important note about
namespaces is that, unlike a python dictionary where the `get` method can be
provided a default value for missing keys, namespaces do not have this
functionality. Therefore, if keys are missing it has the potential to cause
errors. Below is an example of how one could use the namespace for logging
configuration.

> Example provide_namespace
> ```python
> from ngclearn import provide_namespace
> 
> loggerConfig = provide_namespace("logger")
> level = loggerConfig.logging_level
> ```
