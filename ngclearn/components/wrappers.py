from ngclearn.components import sLIFCell
from ngclearn.components import LIFCell

def sLIFWrapper(name, n_units, tau_m, R_m):
    return sLIFCell(name, n_units, tau_m, R_m)

def LIFWrapper(name, n_units, tau_m, R_m):
    return LIFCell(name, n_units, tau_m, R_m)
