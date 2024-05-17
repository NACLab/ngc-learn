import functools

from .sgd import sgd_step, sgd_init
from .adam import adam_step, adam_init

def get_opt_init_fn(opt='adam'):
    return {
        'adam': adam_init,
        'sgd': sgd_init
    }[opt]


def get_opt_step_fn(opt='adam', *args, **kwargs):
    return {
        'adam': functools.partial(adam_step, *args, **kwargs),
        'sgd': functools.partial(sgd_step, *args, **kwargs),
    }[opt]


