from ngcsimlib import JointProcess, MethodProcess
from ngcsimlib.global_state import stateManager
import jax
from typing import TYPE_CHECKING

from ngcsimlib._src.parser.utils import CompiledMethod

if TYPE_CHECKING:
    from ngcsimlib._src.process.baseProcess import BaseProcess

class JaxCompiledMethod(CompiledMethod):
    """
    A wrapper for a compiled method that includes jax's jit wrapped. Used
    exclusively by the mixin and shouldn't be used elsewhere.
    """
    def __init__(self, fn, fn_ast, auxiliary_ast, namespace, extra_globals):
        super().__init__(fn, fn_ast, auxiliary_ast, namespace, extra_globals)
        self._fn = jax.jit(fn)
        self._fn_source = fn

    @property
    def source_fn(self):
        """
        The source method not wrapped in jit
        """
        return self._fn_source

    @classmethod
    def wrap(cls, compiledMethod: CompiledMethod):
        """
        Helper method to expand on a base compiled method
        Args:
            compiledMethod: The method to be expanded upon
        Returns: the JaxCompiledMethod based on the input
        """
        return cls(compiledMethod._fn,
                   compiledMethod.ast,
                   compiledMethod.auxiliary_ast,
                   compiledMethod.namespace,
                   compiledMethod.extra_globals)


class JaxProcessesMixin:
    """
    A mixin for the base Process that adds JAX functionality such as scan and
    implicit jit wrapping
    """
    def __init__(self: "BaseProcess", name, *args, use_jit=True, **kwargs):
        """
        Look at the BaseProcess class for information about other arguments
        Args:
            use_jit:  a flag for if the process should implicitly jit wrap
        """
        super().__init__(name, *args, **kwargs)
        self._previous_result = None
        self._previous_state = None
        self._use_jit = use_jit

    @property
    def previous_result(self):
        """
        Stores and returns the last result of scan (the second returned value)
        """
        return self._previous_result

    @property
    def previous_state(self):
        """
        Stores and returns the last returned state of scan (the first returned
        value)
        """
        return self._previous_state

    def clear(self):
        """
        Clears out the previous result and state from scan
        """
        self._previous_result = None
        self._previous_state = None


    def scan(self: "BaseProcess", inputs, current_state=None, store_state: bool = True, store_results: bool = True):
        """
        Runs the process through jax's scan method
        Args:
            inputs: The inputs for scan (use pack rows to generate), must be a jax array
            current_state: Optional, the current state of the model, if none uses current global state
            store_state: Optional flag, should the final state be stored in the process
            store_results: Optional flag, should the final result be stored in the process

        Returns: the final state, the final result

        """
        state = current_state or stateManager.state
        final_state, result = jax.lax.scan(self.run.compiled, state, inputs)
        if store_state:
            self._previous_state = final_state
        if store_results:
            self._previous_result = result
        return final_state, result

    def compile(self: "baseProcess"):
        """
        For use by the compiler
        """
        super().compile()
        if self._use_jit:
            self.run.compiled = JaxCompiledMethod.wrap(self.run.compiled)


class JaxJointProcess(JaxProcessesMixin, JointProcess):
    pass

class JaxMethodProcess(JaxProcessesMixin, MethodProcess):
    pass
