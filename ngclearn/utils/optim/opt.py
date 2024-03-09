## generic optimizer base/parent class
class Opt():
    """
    A generic base-class for an optimizer.

    Args:
        name: string name of optimizer
    """
    def __init__(self, name):
        self.name = name
        self.time = 0. ## time/step counter

    def update(self, theta, updates): ## apply adjustment to theta
        """
        Apply an update tensor to the current "theta" (parameter) tensor
        according to an internally specified optimization/change rule.

        Args:
            theta: parameter value tensor to change

            updates: externally produced updates to apply to "theta" (note that
                updates should be same shape as "theta" to ensure expected
                behavior)
        """
        pass
