import sys
import subprocess
import pkg_resources
from pkg_resources import get_distribution
from pathlib import Path
from sys import argv

__version__ = get_distribution('ngclearn').version

if not Path(argv[0]).name == "sphinx-build" or Path(argv[0]).name == "build.py":
    required = {'ngclib', 'jax', 'jaxlib'} ## list of core ngclearn dependencies
    installed = {pkg.key for pkg in pkg_resources.working_set}
    missing = required - installed

    for key in required:
        if key in missing:
            raise ImportError(str(key) + ", a core dependency of ngclearn, is not " \
                                "currently installed!")

## Needed to preload is called before anything in ngclearn
import ngclib
from ngclib.controller import Controller

