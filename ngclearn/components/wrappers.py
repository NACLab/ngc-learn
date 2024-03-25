from ngclearn.components import LIFCell

def LIFWrapper(name, n_units, tau_m, R_m):
    return LIFCell(name, n_units, tau_m, R_m)

