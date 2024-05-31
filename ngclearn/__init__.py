import sys
import subprocess
import pkg_resources
from pkg_resources import get_distribution
#from pathlib import Path
#from sys import argv

__version__ = get_distribution('ngclearn').version

if sys.version_info.minor < 10:
    import warnings
    warnings.warn(
        "Running ngclearn and jax in a python version prior to 3.10 may have unintended consequences. Compatability "
        "with python 3.8 is maintained to allow for lava-nc components and should only be used with those")

#required = {'ngcsimlib', 'jax', 'jaxlib'} ## list of core ngclearn dependencies
required = {'ngcsimlib'} #, 'jax', 'jaxlib'}
installed = {pkg.key for pkg in pkg_resources.working_set}
missing = required - installed

for key in required:
    if key in missing:
        raise ImportError(str(key) + ", a core dependency of ngclearn, is not " \
                          "currently installed!")


## Needed to preload is called before anything in ngclearn
from pathlib import Path
from sys import argv
import numpy

import ngcsimlib

from ngcsimlib.context import Context
from ngcsimlib.component import Component
from ngcsimlib.compartment import Compartment
from ngcsimlib.resolver import resolver
from ngcsimlib import utils as sim_utils


from ngcsimlib import configure, preload_modules
from ngcsimlib import logger

if not Path(argv[0]).name == "sphinx-build" or Path(argv[0]).name == "build.py":
    if "readthedocs" not in argv[0]:  ## prevent readthedocs execution of preload
        configure()
        logger.init_logging()
        from ngcsimlib.configManager import get_config
        pkg_config = get_config("packages")
        if pkg_config is not None:
            use_base_numpy = pkg_config.get("use_base_numpy", False)
            if use_base_numpy:
                import numpy as numpy
            else:
                from jax import numpy
        else:
            from jax import numpy


        preload_modules()
