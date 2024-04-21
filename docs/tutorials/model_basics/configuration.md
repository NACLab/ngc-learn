# Lesson 1: Configuring ngcsimlib

## Basics

There are various global configurations that can be made to ngcsimlib, the systems simulation backend for ngc-learn. These include the ability to point to custom locations for the `json_modules` files as well as setting up the logger. In both of these cases, the configuration will generally persist between different models that might be loaded and, thus, it will need to exist outside of the scope of the model's files. To solve this problem, ngcsimlib provides `config.json` as well as the `--config` flag mechanisms.

The `config.json` file contains one large json object with sections set up for different parts of the configuration, broke up into sub-objects. There is no limit to the size or the number of these objects, meaning that the user is free to define and use them as they so choose. However, there are some general design principals that govern ngcsimlib that are worth knowing about. Specifically, this mechanism will not configure any parts of individual models. `config.json` configurations should be used to select/generally set up experiments and control global level flags and not to set hyperparameters for models.

## Built-in Configurations

There are a couple configurations that ngcsimlib will look for while it is initializing. Specifically `modules` and `logging`. While neither of these is needed to get up and running some aspects of ngcsimlib, useful debugging tools such as logging to files and more verbosity are locked behind flags set up here.

### Modules

The modules configuration only contains one value, `module_path`. This value is the location of the `modules.json`, the model-level/experiments-level configuration file one should be setting up when building their experiments. For additional information for configuring this file please
see <a href="../model_basics/json_modules.html">modules.json</a>.

> Example Modules
>
> ```json
> {
>  "modules": {
>    "module_path": "custom/path/to/json/files/modules.json"
>  }
> }
> ```

### Logging

The logging configuration mechanism sets up and controls the instance of the python logger built into ngcsimlib. This mechanism (or JSON section) has three values found within it. Specifically, `logging_level`, `logging_file`, and `hide_console`. The logging levels are the same ones built into the python logger and the value words used are either the standard Python string representation of the level or the numeric equivalent. The `logging file`, if defined, is a file that the logger will append all logging messages to for a more permanent history of all messages. Finally, `hide console`, if set to true, will hide all logging output to the console.

> Default Config
> ```json
> {
>   "logging": {
>     "logging_level": "WARNING",
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

To use a configuration, there are a few options. The first option is to simply use the configuration as a python dictionary. This is done by importing the `get_config` method from `ngcsimlib.configManager` and providing the name of the configuration section to the method.

> Example get_config
>```python
>from ngcsimlib.configManager import get_config
> 
>loggerConfig = get_config("logger")
>level = loggerConfig['logging_level']
>```

The other way you can access a configuration is through a provided namespace. This makes use of python's `SimpleNamespace` to map all the dictionary's key values to properties of an object to be used. One important note about namespaces is that, unlike a python dictionary where the `get` method can be provided a default value for missing keys, namespaces do not have this functionality. Therefore, if keys are missing it has the potential to cause errors. Below is an example of how one could use the namespace for logging configuration.

> Example provide_namespace
> ```python
> from ngcsimlib.configManager import provide_namespace
> 
> loggerConfig = provide_namespace("logger")
> level = loggerConfig.logging_level
> ```
