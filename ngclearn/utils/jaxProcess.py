from ngcsimlib.compilers.process import Process
from jax.lax import scan as _scan
from ngcsimlib.logger import warn
from jax import numpy as jnp

class JaxProcess(Process):
    """
    The JaxProcess is a subclass of the ngcsimlib Process class. The
    functionality added by this subclass is the use of the jax scanner to run a
    process quickly through the use of jax's JIT compiler.
    """
    def scan(self, compartments_to_monitor=None,
             save_state=True, scan_length=None, **kwargs):
        """
        There a quite a few ways to initialize the scan method for the
        jaxProcess. To start the straight forward arguments are
        "compartments_to_monitor" and "save_state". Monitoring compartments
        means at the end of each process cycle record the value of each
        compartment in the list and then at the end a tuple of concatenated
        values will be returned that correspond to each compartment in the
        original list. The save_state flag is simply there to note if the state
        of the model should reflect the final state of the model after the scan
        is complete.

        Where there are options for the arguments is when defining the keyword
        arguments for the process. The process will do its best to broadcast all
        the inputs to the largest size, so they can be scanned over. This means
        that is one is a (2, 3) and the other is a constant, it will broadcast
        constant to a (2, 3). This does mean that every keyword value that is
        passed to a method in the process will be the same size. This is a
        limitation of the jax scanner as all the values have to be concatenated
        into a single jax array to be passed into the scanner. The accepted
        types for arguments, are lists, tuples, numpy arrays, jax arrays, ints,
        and floats. If all the keyword arguments are passed as ints or floats
        the scan_length flag must be set so the scanner knows how many
        iterations to run. If any of the arguments are iterable it will
        automatically assume that the leading axis is the number of iterations
        to run.


        Args:
            compartments_to_monitor: A list of compartments to monitor
            save_state: A boolean flag to indicate if the model state should be
            saved
            scan_length: a value to be used to denote the number of iterations
            of the scanner if all keyword arguments are passed as ints or floats
            **kwargs: the required keyword arguments for the process to run

        Returns: the final state of the model, the stacked output of the scan method

        """
        if compartments_to_monitor is None:
            compartments_to_monitor = []
        arg_order = list(self.get_required_args())

        args = []
        max_axis = 1
        max_next_axis = 0

        for kwarg in arg_order:
            if kwarg not in kwargs.keys():
                warn("Missing kwarg in Process", self.name)
                return

            kval = kwargs.get(kwarg, None)
            if isinstance(kval, (float, int, list, tuple)):
                val = jnp.array(kval)
            else:
                val = kval

            max_axis = max(max_axis, len(val.shape))
            if max_axis == len(val.shape):
                max_next_axis = max(max_next_axis, val.shape[0])
            args.append(val)

        # Check axis && get max_next_axis

        if max_next_axis == 0:
            if scan_length is None:
                warn("scan_length must be defined if all keyword arguments are "
                     "constants")
                return
            elif scan_length > 0:
                max_next_axis = scan_length
            else:
                warn("scan_length must be greater than 0")
                return

        for axis in range(max_axis):
            current_axis = max_next_axis
            max_next_axis = 0
            new_args = []
            for a in args:
                if len(a.shape) >= axis+1:
                    if a.shape[axis] == current_axis:
                        new_args.append(a)
                    else:
                        warn("Keyword arguments must all be able to be "
                             "broadcasted to the largest shape")
                        return
                else:
                    new_args.append(jnp.zeros(list(a.shape) + [current_axis], dtype=a.dtype) + a.reshape(*a.shape, 1))

                if len(a.shape) > axis+1:
                    max_next_axis = max(max_next_axis, a.shape[axis+1])

            args = new_args

        args = jnp.array(args).transpose([1, 0] + [i for i in range(2, max_axis+1)])

        def _pure(current_state, x):
            v = self.pure(current_state, **{key: value for key, value in zip(arg_order, x)})
            return v, [v[c.path] for c in compartments_to_monitor]

        vals, stacked = _scan(_pure, init=self.get_required_state(include_special_compartments=True), xs=args)
        if save_state:
            self.updated_modified_state(vals)
        return vals, stacked
