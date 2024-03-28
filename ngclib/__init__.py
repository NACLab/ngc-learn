from . import utils
from . import controller
from . import commands

import argparse, os, warnings, json
from types import SimpleNamespace
from pathlib import Path
from sys import argv
from importlib import import_module

from pkg_resources import get_distribution

__version__ = get_distribution('ngclib').version ## set software version

###### Preload Modules
def preload():
    parser = argparse.ArgumentParser(description='Build and run a model using ngclean')
    parser.add_argument("--modules", type=str, help='location of modules.json file')

    args = parser.parse_args()
    try:
        module_path = args.modules
    except:
        module_path = None

    if module_path is None:
        module_path = "json_files/modules.json"

    if not os.path.isfile(module_path):
        warnings.warn("Missing file to preload modules from. Attempted to locate file at \"" + str(module_path) + "\"" )
        return

    with open(module_path, 'r') as file:
        modules = json.load(file, object_hook=lambda d: SimpleNamespace(**d))

    for module in modules:
        mod = import_module(module.absolute_path)
        utils._Loaded_Modules[module.absolute_path] = mod

        for attribute in module.attributes:
            atr = getattr(mod, attribute.name)
            utils._Loaded_Attributes[attribute.name] = atr

            utils._Loaded_Attributes[".".join([module.absolute_path, attribute.name])] = atr
            if hasattr(attribute, "keywords"):
                for keyword in attribute.keywords:
                    utils._Loaded_Attributes[keyword] = atr

if not Path(argv[0]).name == "sphinx-build" or Path(argv[0]).name == "build.py":
    preload()
