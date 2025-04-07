from ngcsimlib.compilers.process import Process
from jax.lax import scan as _scan


class JaxProcess(Process):
    def scan(self, xs, arg_order=None, compartments_to_monitor=None, save_state=True):
        if compartments_to_monitor is None:
            compartments_to_monitor = []
        if arg_order is None:
            arg_order = list(self.get_required_args())

        def _pure(current_state, x):
            v = self.pure(current_state, **{key: value for key, value in zip(arg_order, x)})
            return v, [v[c.path] for c in compartments_to_monitor]

        vals, stacked = _scan(_pure, init=self.get_required_state(include_special_compartments=True), xs=xs)
        if save_state:
            self.updated_modified_state(vals)
        return stacked
    