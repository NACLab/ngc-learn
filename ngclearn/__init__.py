import sys
import subprocess
import pkg_resources
from pkg_resources import get_distribution

__version__ = get_distribution('ngclearn').version

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

