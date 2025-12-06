from ngcsimlib import JointProcess, MethodProcess
from ngcsimlib.global_state import stateManager
import jax
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ngcsimlib._src.process.baseProcess import BaseProcess

class JaxProcessesMixin:
    def __init__(self: "BaseProcess"):
        self._previous_result = None
        self._previous_state = None

    @property
    def previous_result(self):
        return self._previous_result

    @property
    def previous_state(self):
        return self._previous_state

    def clear(self):
        self._previous_result = None
        self._previous_state = None


    def scan(self: "BaseProcess", inputs, current_state=None, save_state: bool = True, store_results: bool = True):
        state = current_state or stateManager.state
        final_state, result = jax.lax.scan(self.run.compiled, state, inputs)
        if save_state:
            self._previous_state = final_state
        if store_results:
            self._previous_result = result
        return final_state, result



class JaxJointProcess(JointProcess, JaxProcessesMixin):
    pass

class JaxMethodProcess(MethodProcess, JaxProcessesMixin):
    pass
