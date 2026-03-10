from collections.abc import Callable
from typing import Protocol, TypeVar, runtime_checkable

T = TypeVar("T")


@runtime_checkable
class Crossover(Protocol[T]):
    """Protocol for crossover operations in evolutionary algorithms."""

    def __call__(self, parents: list[T]) -> list[T]:
        """
        Combine parents to create offspring.

        Args:
            parents: A list of parent individuals selected for reproduction.

        Returns:
            list[T]: A list of new offspring individuals produced by the crossover.
        """
        ...


def crossover_fn(fn: Callable[[list[T]], list[T]]) -> Crossover[T]:
    """
    Decorator to convert a function into a Crossover protocol implementation.

    Args:
        fn: A function that takes a list of parents and returns a list of offspring.

    Returns:
        Crossover[T]: An object implementing the Crossover protocol.
    """

    class Wrapper:
        def __init__(self, func: Callable[[list[T]], list[T]]):
            self.func = func

        def __call__(self, parents: list[T]) -> list[T]:
            return self.func(parents)

    return Wrapper(fn)  # type: ignore[return-value]
