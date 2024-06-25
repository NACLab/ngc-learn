from ngclearn.components.base_monitor import Base_Monitor


class Monitor(Base_Monitor):
    """
    A numpy implementation of `Base_Monitor`. Designed to be used with all lava compatible ngclearn components
    """

    @staticmethod
    def build_advance(compartments):
        @staticmethod
        def _advance(**kwargs):
            return_vals = []
            for comp in compartments:
                new_val = kwargs[comp]
                current_store = kwargs[comp + "*store"]
                current_store[:-1] = current_store[1:]
                current_store[-1] = new_val
                return_vals.append(current_store)
            return return_vals if len(compartments) > 1 else return_vals[0]

        return _advance
