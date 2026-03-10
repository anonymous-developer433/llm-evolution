from collections.abc import Callable
from typing import Protocol, TypeVar, runtime_checkable

T = TypeVar("T")


@runtime_checkable
class Selection(Protocol[T]):
    """Protocol for selecting survivors for the next generation."""

    def __call__(
        self, population: list[T], offspring: list[T], fitness_scores: list[float]
    ) -> list[T]:
        """
        Select survivors for the next generation from the combined pool of current population and offspring.

        Args:
            population: The current population of individuals.
            offspring: The new offspring produced in the current generation.
            fitness_scores: The fitness scores corresponding to the combined individuals (population + offspring).

        Returns:
            list[T]: The list of individuals selected to survive to the next generation.
        """
        ...


def selection_fn(
    fn: Callable[[list[T], list[T], list[float]], list[T]],
) -> Selection[T]:
    """
    Decorator to convert a function into a Selection protocol implementation.

    Args:
        fn: A function that takes population, offspring, and fitness scores and returns survivors.

    Returns:
        Selection[T]: An object implementing the Selection protocol.
    """

    class Wrapper:
        def __init__(self, func: Callable[[list[T], list[T], list[float]], list[T]]):
            self.func = func

        def __call__(
            self, population: list[T], offspring: list[T], fitness_scores: list[float]
        ) -> list[T]:
            return self.func(population, offspring, fitness_scores)

    return Wrapper(fn)  # type: ignore[return-value]
