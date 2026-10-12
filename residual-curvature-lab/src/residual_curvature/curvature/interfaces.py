"""Framework-neutral contracts; implementations may use PyTorch or JAX arrays."""
from typing import Any, Protocol


class LinearOperator(Protocol):
    def __call__(self, vector: Any) -> Any: ...


class Metric(Protocol):
    def apply(self, vector: Any) -> Any: ...
    def inverse(self, vector: Any) -> Any: ...
