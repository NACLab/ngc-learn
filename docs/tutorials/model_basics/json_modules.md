# Lesson 2: Configuring with the modules.json File

## Basic Usage:

The basic usage for the `modules.json` file is to provide ngclearn with a list of modules to import and associated
classes that are needed to build the models it will be loading. If there is a need to use the imported
modules outside of these cases, use `ngcsimlib.utils.load_attribute` and the loaded
attribute will be returned.

By default, <a href="https://github.com/NACLab/ngc-sim-lib">ngcsimlib</a>, the backend
dependency of ngc-learn, looks for `json_files/modules.json` in your project path.
However, this can be changed inside the
<a href=https://ngc-learn.readthedocs.io/en/latest/tutorials/model_basics/configuration.html>configuration file</a>. In
the event that this
file is missing, ngcsimlib will not break but its ability to load saved models will be limited.

## Motivation

The motivation behind the use of `modules.json` versus the registering all the
various parts of the model at the top of the file is reusability. When all the
parts have to be registered/imported at the top of every test file, or be placed into specific locations can be limiting
and slows down development. With a single project wide modules file all loaded models can look there to load components.
This also allows for components to be saved in humanreadable formats not as a pickled object as we can save and load all
the relevant class information from the class name and the modules file.

## Structure

A complete schema for the modules file can be found in `modules.schema`

The general structure of the modules file can be thought of as a transformation
of python import statements to JSON objects. Take the following example:

```python
from ngclearn.commands import AdvanceState as advance
```

In this statement we are importing a command from ngcsimlib and aliasing it to the
word "advance". Now we will transform this into JSON for the modules file. First,
we take the top level module that we are importing from, in this case
`ngcsimlib.commands`; this the absolute path to the location of this module. Next,
we look at the name of what we are importing here: `AdvanceState`. Finally, we
look at the keyword since this import is being assigned to `advance`. We then
take these three parts and combine them into the following JSON object:

```json
  {
  "absolute_path": "ngclearn.commands",
  "attributes": [
    {
      "name": "AdvanceState",
      "keywords": [
        "advance"
      ]
    }
  ]
}
```

Now there are a few additional things that this JSON formulation of an import
allows us to do. Primarily, it allows for multiple keywords for a single import
to be defined. This if we wanted to use `advance` and `adv` all we would do is
change the keyword line to `"keywords": ["advance", "adv"]`. In addition, we are able
to specify more than one attribute to import from a single top level module
such as also importing the evolve command.

```json
  {
  "absolute_path": "ngcsimlib.commands",
  "attributes": [
    {
      "name": "AdvanceState",
      "keywords": [
        "advance",
        "adv"
      ]
    },
    {
      "name": "Evolve"
    }
  ]
}
```

Now you might notice above that, when importing the evolve attribute, no
keywords were given. This means that, in order to add an evolve command to
the controller, the whole name will need to be given. There is one caveat to
this scheme though; it is case-insensitive by default, meaning that both
`Evolve` and `evolve` are valid ways to using this import.

## Example Transformations

Below are some additional examples to help with transitioning from python
header import statements to JSON configuration.

> Case 1
> Python:
> ```python
> from ngcsimlib.commands import AdvanceState as advance, Evolve, Multiclamp as mClamp
> ```
> Json:
> ```json
> [
>   {
>     "absolute_path": "ngcsimlib.commands",
>     "attributes": [
>       {
>         "name": "AdvanceState",
>         "keywords": ["advance"]
>       },
>       {
>         "name": "Evolve"
>       },
>       {
>         "name": "Multiclamp",
>         "keywords": "mClamp"
>       }
>     ]
>   }
> ]
> ```

> Case 2
> Python
> ```python
> from ngclearn.commands import AdvanceState as advance
> from ngclearn.operations import summation as summ, overwrite
> ```
>
> Json
> ```json
> [
>   {
>     "absolute_path": "ngclearn.commands",
>     "attributes": [
>       {
>         "name": "AdvanceState",
>         "keywords": ["advance"]
>       }
>     ]
>   },
>   {
>     "absolute_path": "ngclearn.operations",
>     "attributes": [
>       {
>         "name": "summation",
>         "keywords": ["summ"]
>       },
>       {
>         "name": "overwrite"
>       }
>     ]
>   }
> ]
> ```
