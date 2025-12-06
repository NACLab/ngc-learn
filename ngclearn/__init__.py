import sys
import pkg_resources
from pkg_resources import get_distribution

__version__ = get_distribution('ngclearn').version

if sys.version_info.minor < 10:
    import warnings
    warnings.warn(
        "Running ngclearn and jax in a python version prior to 3.10 may have unintended consequences. Compatibility "
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

from ngclearn.utils import JointProcess, MethodProcess
from ngcsimlib.context import Context, ContextObjectTypes
from ngcsimlib import Component
from ngcsimlib.compartment import Compartment
from ngcsimlib import logger, get_config, provide_namespace
from ngcsimlib.parser import compilable
from ngcsimlib.operations import Summation, Product


if not Path(argv[0]).name == "sphinx-build" or Path(argv[0]).name == "build.py":
    if "readthedocs" not in argv[0]:  ## prevent readthedocs execution of preload
        from ngcsimlib import configure
        configure()
        logger.init_logging()
