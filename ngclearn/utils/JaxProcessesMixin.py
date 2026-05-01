from ngcsimlib import JointProcess, MethodProcess
from ngcsimlib.global_state import stateManager
import jax
from typing import TYPE_CHECKING

from ngcsimlib._src.parser.utils import CompiledMethod

if TYPE_CHECKING:
    from ngcsimlib._src.process.baseProcess import BaseProcess

class JaxCompiledMethod(CompiledMethod):
    def __init__(self, fn, fn_ast, auxiliary_ast, namespace, extra_globals):
        super().__init__(fn, fn_ast, auxiliary_ast, namespace, extra_globals)
        self._fn = jax.jit(fn)
        self._fn_source = fn

    @property
    def source_fn(self):
        return self._fn_source

    @classmethod
    def wrap(cls, compiledMethod: CompiledMethod):
        return cls(compiledMethod._fn,
                   compiledMethod.ast,
                   compiledMethod.auxiliary_ast,
                   compiledMethod.namespace,
                   compiledMethod.extra_globals)


class JaxProcessesMixin:
    def __init__(self: "BaseProcess", name, *args, use_jit=True, **kwargs):
        super().__init__(name, *args, **kwargs)
        self._previous_result = None
        self._previous_state = None
        self._use_jit = use_jit

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

    def compile(self: "baseProcess"):
        super().compile()
        if self._use_jit:
            self.run.compiled = JaxCompiledMethod.wrap(self.run.compiled)


class JaxJointProcess(JaxProcessesMixin, JointProcess):
    pass

class JaxMethodProcess(JaxProcessesMixin, MethodProcess):
    pass
