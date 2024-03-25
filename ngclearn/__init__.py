import sys
import subprocess
import pkg_resources
from pkg_resources import get_distribution

__version__ = get_distribution('ngclearn').version

required = {'ngclib'}
installed = {pkg.key for pkg in pkg_resources.working_set}
missing = required - installed
if len(missing) >= 1:
    print("Error: ngclib, a core dependency of ngclearn, is not currently installed!")
    sys.exit(1)
    ## below will auto-install missing dependency silently (AO: not using this yet)
    #python = sys.executable
    #subprocess.check_call([python, '-m', 'pip', 'install', *missing], stdout=subprocess.DEVNULL)
