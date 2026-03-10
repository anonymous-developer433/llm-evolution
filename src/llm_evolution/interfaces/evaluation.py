from collections.abc import Callable
from typing import Protocol, TypeVar, runtime_checkable

T = TypeVar("T", contravariant=True)
T_fn = TypeVar("T_fn")


@runtime_checkable
class Evaluation(Protocol[T]):
    """Protocol for evaluating the fitness of an instance."""

    def __call__(self, instance: T) -> float:
        """
        Calculate and return the fitness score for a given individual.

        Args:
            instance: The individual instance to evaluate.

        Returns:
            float: The calculated fitness score. Higher scores typically represent better individuals.
        """
        ...


def evaluation_fn(fn: Callable[[T_fn], float]) -> Evaluation[T_fn]:
    """
    Decorator to convert a function into an Evaluation protocol implementation.

    Args:
        fn: A function that takes an instance and returns its fitness score.

    Returns:
        Evaluation[T]: An object implementing the Evaluation protocol.
    """

    class Wrapper:
        def __init__(self, func: Callable[[T_fn], float]):
            self.func = func

        def __call__(self, instance: T_fn) -> float:
            return self.func(instance)

    return Wrapper(fn)  # type: ignore[return-value]
