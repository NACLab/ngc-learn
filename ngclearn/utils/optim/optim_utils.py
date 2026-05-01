import functools
from .sgd import sgd_step, sgd_init
from .nag import nag_step, nag_init
from .adam import adam_step, adam_init

def get_opt_init_fn(opt='adam'):
    return {
        'adam': adam_init,
        'nag': nag_init,
        'sgd': sgd_init
    }[opt]


def get_opt_step_fn(opt='adam', **kwargs):
    ## **kwargs here is the hyper-parameters you want to pass in the optimization function
    return {
        'adam': functools.partial(adam_step, **kwargs),
        'nag': functools.partial(nag_step, **kwargs),
        'sgd': functools.partial(sgd_step, **kwargs),
    }[opt]
