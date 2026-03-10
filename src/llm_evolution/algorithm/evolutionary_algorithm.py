import logging
import random
from dataclasses import dataclass
from pickle import PicklingError
from typing import Generic, TypeVar

from loky import get_reusable_executor

from ..interfaces.crossover import Crossover
from ..interfaces.evaluation import Evaluation
from ..interfaces.finish_condition import FinishCondition
from ..interfaces.initial_population import InitialPopulation
from ..interfaces.mutation import Mutation
from ..interfaces.selection import Selection

T = TypeVar("T")

logger = logging.getLogger(__name__)


@dataclass
class EvolutionResult(Generic[T]):
    """
    Result of the evolutionary algorithm.

    Attributes:
        best_instance: The best individual found during evolution.
        best_fitness: The fitness score of the best individual.
        population: The final population after evolution.
        generation: The number of generations executed.
    """

    best_instance: T
    best_fitness: float
    population: list[T]
    generation: int


class EvolutionaryAlgorithm(Generic[T]):
    """
    Standard Evolutionary Algorithm implementation.

    This class orchestrates the evolutionary process by coordinating
    initialization, evaluation, crossover, mutation, and selection.

    Attributes:
        initial_population: Strategy to generate the initial population.
        evaluation: Strategy to evaluate individual fitness.
        selection: Strategy to select survivors for the next generation.
        finish_condition: Strategy to determine when evolution should stop.
        crossover: Optional strategy for crossover operations.
        mutation: Optional strategy for mutation operations.
        population_size: The number of individuals in the population.
    """

    def __init__(
        self,
        initial_population: InitialPopulation[T],
        evaluation: Evaluation[T],
        selection: Selection[T],
        finish_condition: FinishCondition[T],
        crossover: Crossover[T] | None = None,
        mutation: Mutation[T] | None = None,
        population_size: int = 100,
        max_workers: int | None = None,
    ):
        """
        Initialize the evolutionary algorithm.

        Args:
            initial_population: Strategy to generate the initial population.
            evaluation: Strategy to evaluate individual fitness.
            selection: Strategy to select survivors for the next generation.
            finish_condition: Strategy to determine when evolution should stop.
            crossover: Optional strategy for crossover operations.
            mutation: Optional strategy for mutation operations.
            population_size: The number of individuals in the population.
            max_workers: Maximum number of parallel workers for crossover and mutation.
                If None, defaults to the number of CPUs.
        """
        self.initial_population = initial_population
        self.evaluation = evaluation
        self.selection = selection
        self.finish_condition = finish_condition
        self.crossover = crossover
        self.mutation = mutation
        self.population_size = population_size
        self.max_workers = max_workers
        self._fitness_cache: dict[int, float] = {}

    def _evaluate_individuals(self, individuals: list[T]) -> list[float]:
        """
        Evaluate individuals using the cache to avoid redundant computations.

        Only individuals not already in the cache are evaluated. Results are
        stored in the cache for future lookups.

        Args:
            individuals: The list of individuals to evaluate.

        Returns:
            list[float]: The fitness scores in the same order as the input.
        """
        results: list[float] = []
        for ind in individuals:
            obj_id = id(ind)
            if obj_id not in self._fitness_cache:
                self._fitness_cache[obj_id] = self.evaluation(ind)
            results.append(self._fitness_cache[obj_id])
        return results

    def _prune_cache(self, alive: list[T]) -> None:
        """
        Remove cache entries for individuals no longer referenced.

        Args:
            alive: The list of individuals that should remain in the cache.
        """
        alive_ids = {id(ind) for ind in alive}
        self._fitness_cache = {
            k: v for k, v in self._fitness_cache.items() if k in alive_ids
        }

    def _parallel_crossover(self, parents_list: list[list[T]]) -> list[T]:
        """
        Execute crossover operations in parallel using loky.

        Falls back to sequential execution if the callable cannot be pickled.

        Args:
            parents_list: A list of parent pairs for crossover.

        Returns:
            list[T]: The combined offspring from all crossover operations.
        """
        try:
            executor = get_reusable_executor(max_workers=self.max_workers)
            futures = [
                executor.submit(self.crossover, parents) for parents in parents_list
            ]
            offspring: list[T] = []
            for future in futures:
                offspring.extend(future.result())
            return offspring
        except PicklingError:
            logger.warning(
                "Crossover callable could not be pickled. Falling back to sequential execution."
            )
            return self._sequential_crossover(parents_list)

    def _sequential_crossover(self, parents_list: list[list[T]]) -> list[T]:
        """
        Execute crossover operations sequentially.

        Args:
            parents_list: A list of parent pairs for crossover.

        Returns:
            list[T]: The combined offspring from all crossover operations.
        """
        assert self.crossover is not None
        offspring: list[T] = []
        for parents in parents_list:
            offspring.extend(self.crossover(parents))
        return offspring

    def _parallel_mutation(self, to_mutate: list[T]) -> list[T]:
        """
        Execute mutation operations in parallel using loky.

        Falls back to sequential execution if the callable cannot be pickled.

        Args:
            to_mutate: The list of individuals to mutate.

        Returns:
            list[T]: The list of successfully mutated individuals (non-None results).
        """
        try:
            executor = get_reusable_executor(max_workers=self.max_workers)
            futures = [executor.submit(self.mutation, ind) for ind in to_mutate]
            mutated: list[T] = []
            for future in futures:
                result = future.result()
                if result is not None:
                    mutated.append(result)
            return mutated
        except PicklingError:
            logger.warning(
                "Mutation callable could not be pickled. Falling back to sequential execution."
            )
            return self._sequential_mutation(to_mutate)

    def _sequential_mutation(self, to_mutate: list[T]) -> list[T]:
        """
        Execute mutation operations sequentially.

        Args:
            to_mutate: The list of individuals to mutate.

        Returns:
            list[T]: The list of successfully mutated individuals (non-None results).
        """
        assert self.mutation is not None
        mutated: list[T] = []
        for ind in to_mutate:
            result = self.mutation(ind)
            if result is not None:
                mutated.append(result)
        return mutated

    def run(self, log: bool = False) -> EvolutionResult[T]:
        """
        Execute the evolutionary algorithm.

        Args:
            log: Whether to enable logging of the evolutionary process.

        Returns:
            EvolutionResult: The result containing the best individual and final population.
        """
        if log:
            logging.basicConfig(level=logging.INFO)

        if log:
            logger.info(
                "Starting evolutionary algorithm with population size %d",
                self.population_size,
            )

        self._fitness_cache = {}
        population = self.initial_population(self.population_size)
        generation = 0

        while True:
            fitness_scores = self._evaluate_individuals(population)
            best_fitness = max(fitness_scores)

            if log:
                logger.info(
                    "Generation %d: Best fitness = %.4f", generation, best_fitness
                )

            if self.finish_condition(population, generation, fitness_scores):
                if log:
                    logger.info("Finish condition met at generation %d", generation)
                break

            offspring: list[T] = []

            if self.crossover and len(population) >= 2:
                parents_list = [
                    random.sample(population, 2)
                    for _ in range(self.population_size // 2)
                ]
                offspring.extend(self._parallel_crossover(parents_list))

            if self.mutation:
                to_mutate = random.sample(
                    population + offspring,
                    min(len(population), len(population) + len(offspring)),
                )
                offspring.extend(self._parallel_mutation(to_mutate))

            offspring_fitness = self._evaluate_individuals(offspring)
            combined_fitness = fitness_scores + offspring_fitness

            population = self.selection(population, offspring, combined_fitness)
            generation += 1

            if len(population) > self.population_size:
                pop_fitness = self._evaluate_individuals(population)
                indexed_fitness = list(enumerate(pop_fitness))
                indexed_fitness.sort(key=lambda x: x[1], reverse=True)
                population = [
                    population[i] for i, _ in indexed_fitness[: self.population_size]
                ]

            self._prune_cache(population)

        final_fitness = self._evaluate_individuals(population)
        best_idx = final_fitness.index(max(final_fitness))

        if log:
            logger.info(
                "Evolution finished. Best fitness: %.4f", final_fitness[best_idx]
            )

        return EvolutionResult(
            best_instance=population[best_idx],
            best_fitness=final_fitness[best_idx],
            population=population,
            generation=generation,
        )
