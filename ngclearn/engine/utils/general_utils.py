"""
General/arbitrary utilities.
"""

class ParamSeq:
    def __init__(self, params: dict, constant_keys: list | None =None):
        if constant_keys is None:
            constant_keys = []

        self.constant_keys = constant_keys
        self.sequence_keys = []
        self.params = params

        for key in params.keys():
            if key not in constant_keys:
                if hasattr(params[key], '__iter__'):
                    self.sequence_keys.append(key)
                else:
                    self.constant_keys.append(key)

        self.sequence_length = 0
        if len(self.sequence_keys) == 0:
            self.sequence_length = 1
        else:
            self.sequence_length = len(self.params[self.sequence_keys[0]])

        self.idx = 0

    def __iter__(self):
        self.idx = 0
        return self

    def __next__(self):
        if self.idx < self.sequence_length:
            params = {}
            for key in self.constant_keys:
                params[key] = self.params[key]
            for key in self.sequence_keys:
                params[key] = self.params[key][self.idx]
            self.idx += 1
            return params
        raise StopIteration

    def reset(self):
        self.idx = 0
