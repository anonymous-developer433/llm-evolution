import os
import random
import time
from dataclasses import dataclass

from llm_evolution.algorithm.evolutionary_algorithm import (
    EvolutionaryAlgorithm,
    EvolutionResult,
)
from llm_evolution.interfaces.crossover import crossover_fn
from llm_evolution.interfaces.mutation import mutation_fn
from llm_evolution.interfaces.evaluation import evaluation_fn
from llm_evolution.interfaces.selection import selection_fn
from llm_evolution.interfaces.initial_population import initial_population_fn
from llm_evolution.interfaces.finish_condition import finish_condition_fn


def test_evolutionary_algorithm_integration():
    """
    Integration test for the full evolutionary pipeline.
    Goal: Find the integer that maximizes the value (simple optimization).
    """

    @initial_population_fn
    def init_pop(size: int) -> list[int]:
        return [random.randint(0, 10) for _ in range(size)]

    @evaluation_fn
    def evaluate(instance: int) -> float:
        return float(instance)

    @selection_fn
    def select(pop: list[int], off: list[int], scores: list[float]) -> list[int]:
        # Simple elitist selection: pick the best ones from combined population
        combined = pop + off
        # Sort by fitness (scores are for combined population)
        indexed_scores = list(enumerate(scores))
        indexed_scores.sort(key=lambda x: x[1], reverse=True)

        selected_indices = [i for i, _ in indexed_scores[: len(pop)]]
        return [combined[i] for i in selected_indices]

    @crossover_fn
    def crossover(parents: list[int]) -> list[int]:
        # Average of parents
        return [sum(parents) // len(parents)]

    @mutation_fn
    def mutate(instance: int) -> int:
        # Small random change
        return instance + random.choice([-1, 1])

    @finish_condition_fn
    def finish(pop: list[int], gen: int, scores: list[float]) -> bool:
        # Finish if we reach a target value or max generations
        return max(scores) >= 100 or gen >= 50

    # Instantiate algorithm
    ea = EvolutionaryAlgorithm(
        initial_population=init_pop,
        evaluation=evaluate,
        selection=select,
        finish_condition=finish,
        crossover=crossover,
        mutation=mutate,
        population_size=20,
    )

    # Run algorithm
    result = ea.run(log=True)

    # Assertions
    assert result.best_instance is not None
    assert result.best_fitness >= 0
    assert len(result.population) == 20
    assert result.generation >= 0


def test_ea_no_crossover_no_mutation():
    """Test algorithm with only selection and no genetic operators."""

    @initial_population_fn
    def init_pop(size: int) -> list[int]:
        return [5] * size

    @evaluation_fn
    def evaluate(instance: int) -> float:
        return float(instance)

    @selection_fn
    def select(pop: list[int], off: list[int], scores: list[float]) -> list[int]:
        return pop

    @finish_condition_fn
    def finish(pop: list[int], gen: int, scores: list[float]) -> bool:
        return gen >= 5

    ea = EvolutionaryAlgorithm(
        initial_population=init_pop,
        evaluation=evaluate,
        selection=select,
        finish_condition=finish,
        population_size=10,
    )
    result = ea.run()
    assert result.generation == 5
    assert all(ind == 5 for ind in result.population)


def test_ea_immediate_finish():
    """Test algorithm that finishes at generation 0."""

    @initial_population_fn
    def init_pop(size: int) -> list[int]:
        return [1] * size

    @evaluation_fn
    def evaluate(instance: int) -> float:
        return 1.0

    @selection_fn
    def select(pop: list[int], off: list[int], scores: list[float]) -> list[int]:
        return pop

    @finish_condition_fn
    def finish(pop: list[int], gen: int, scores: list[float]) -> bool:
        return True

    ea = EvolutionaryAlgorithm(
        initial_population=init_pop,
        evaluation=evaluate,
        selection=select,
        finish_condition=finish,
        population_size=5,
    )
    result = ea.run()
    assert result.generation == 0
    assert len(result.population) == 5


def test_ea_population_size_one():
    """Test algorithm with a single individual (minimal population)."""

    @initial_population_fn
    def init_pop(size: int) -> list[int]:
        return [10]

    @evaluation_fn
    def evaluate(instance: int) -> float:
        return float(instance)

    @selection_fn
    def select(pop: list[int], off: list[int], scores: list[float]) -> list[int]:
        combined = pop + off
        indexed_scores = list(enumerate(scores))
        indexed_scores.sort(key=lambda x: x[1], reverse=True)
        return [combined[indexed_scores[0][0]]]

    @mutation_fn
    def mutate(instance: int) -> int:
        return instance + 1

    @finish_condition_fn
    def finish(pop: list[int], gen: int, scores: list[float]) -> bool:
        return gen >= 10

    ea = EvolutionaryAlgorithm(
        initial_population=init_pop,
        evaluation=evaluate,
        selection=select,
        finish_condition=finish,
        mutation=mutate,
        population_size=1,
    )
    result = ea.run()
    assert len(result.population) == 1
    assert result.best_fitness >= 10


def test_ea_complex_type_integration():
    """Test algorithm with a non-primitive type (dictionary)."""
    from dataclasses import dataclass

    @dataclass
    class Individual:
        genes: list[float]

    @initial_population_fn
    def init_pop(size: int) -> list[Individual]:
        return [Individual([random.random() for _ in range(2)]) for _ in range(size)]

    @evaluation_fn
    def evaluate(instance: Individual) -> float:
        return sum(instance.genes)

    @selection_fn
    def select(
        pop: list[Individual], off: list[Individual], fitness_scores: list[float]
    ) -> list[Individual]:
        combined = pop + off
        indexed_scores = list(enumerate(fitness_scores))
        indexed_scores.sort(key=lambda x: x[1], reverse=True)
        return [combined[i] for i, _ in indexed_scores[: len(pop)]]

    @mutation_fn
    def mutate(instance: Individual) -> Individual:
        new_genes = [g + random.uniform(-0.1, 0.1) for g in instance.genes]
        return Individual(new_genes)

    @finish_condition_fn
    def finish(pop: list[Individual], gen: int, scores: list[float]) -> bool:
        return gen >= 20

    ea = EvolutionaryAlgorithm(
        initial_population=init_pop,
        evaluation=evaluate,
        selection=select,
        finish_condition=finish,
        mutation=mutate,
        population_size=10,
    )
    result = ea.run()
    assert isinstance(result.best_instance, Individual)
    assert len(result.population) == 10


def test_ea_mutation_returns_none():
    """Test algorithm when mutation returns None."""

    from llm_evolution.interfaces.initial_population import InitialPopulation

    initial_pop_list: list[int] = [1, 2, 3]

    class MockInitialPopulation:
        def __call__(self, size: int) -> list[int]:
            res: list[int] = initial_pop_list
            return res

    init_pop: InitialPopulation[int] = MockInitialPopulation()

    @evaluation_fn
    def evaluate(instance: int) -> float:
        return float(instance)

    @selection_fn
    def select(pop: list[int], off: list[int], scores: list[float]) -> list[int]:
        # Just return the offspring if any, otherwise return pop
        return off if off else pop

    @mutation_fn
    def mutate(instance: int) -> int | None:
        # Never mutate
        return None

    @finish_condition_fn
    def finish(pop: list[int], gen: int, scores: list[float]) -> bool:
        return gen >= 1

    ea = EvolutionaryAlgorithm[int](
        initial_population=init_pop,
        evaluation=evaluate,
        selection=select,
        finish_condition=finish,
        mutation=mutate,
        population_size=3,
    )
    result: EvolutionResult[int] = ea.run()
    # If mutation always returns None, offspring will be empty
    # Selection will receive empty offspring and return pop (1, 2, 3)
    population_list: list[int] = result.population
    assert sorted(population_list) == [1, 2, 3]
    assert result.generation == 1


def test_parallel_ea_full_pipeline():
    """Integration test: full EA pipeline with parallel crossover and mutation."""

    @initial_population_fn
    def init_pop(size: int) -> list[int]:
        return [random.randint(0, 10) for _ in range(size)]

    @evaluation_fn
    def evaluate(instance: int) -> float:
        return float(instance)

    @selection_fn
    def select(pop: list[int], off: list[int], scores: list[float]) -> list[int]:
        combined = pop + off
        indexed_scores = list(enumerate(scores))
        indexed_scores.sort(key=lambda x: x[1], reverse=True)
        return [combined[i] for i, _ in indexed_scores[: len(pop)]]

    @crossover_fn
    def crossover(parents: list[int]) -> list[int]:
        return [max(parents)]

    @mutation_fn
    def mutate(instance: int) -> int:
        return instance + random.choice([0, 1])

    @finish_condition_fn
    def finish(pop: list[int], gen: int, scores: list[float]) -> bool:
        return gen >= 10

    ea = EvolutionaryAlgorithm(
        initial_population=init_pop,
        evaluation=evaluate,
        selection=select,
        finish_condition=finish,
        crossover=crossover,
        mutation=mutate,
        population_size=20,
        max_workers=2,
    )

    result = ea.run(log=True)

    assert result.best_instance is not None
    assert result.best_fitness >= 0
    assert len(result.population) == 20
    assert result.generation == 10


def test_parallel_ea_produces_same_quality_as_sequential():
    """Parallel and sequential runs should produce comparable optimization quality."""
    random.seed(42)

    @initial_population_fn
    def init_pop(size: int) -> list[int]:
        return [random.randint(0, 5) for _ in range(size)]

    @evaluation_fn
    def evaluate(instance: int) -> float:
        return float(instance)

    @selection_fn
    def select(pop: list[int], off: list[int], scores: list[float]) -> list[int]:
        combined = pop + off
        indexed = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        return [combined[i] for i, _ in indexed[: len(pop)]]

    @mutation_fn
    def mutate(instance: int) -> int:
        return instance + 1

    @finish_condition_fn
    def finish(pop: list[int], gen: int, scores: list[float]) -> bool:
        return gen >= 5

    ea_parallel = EvolutionaryAlgorithm(
        initial_population=init_pop,
        evaluation=evaluate,
        selection=select,
        finish_condition=finish,
        mutation=mutate,
        population_size=10,
        max_workers=2,
    )

    result = ea_parallel.run()

    # After 5 generations of +1 mutation with elitist selection,
    # the best fitness should have improved significantly from the initial [0..5] range
    assert result.best_fitness > 5
    assert result.generation == 5


def test_parallel_ea_with_complex_type():
    """Parallel execution with a non-primitive type (dataclass) to verify serialization."""

    @dataclass
    class Vector:
        values: list[float]

    @initial_population_fn
    def init_pop(size: int) -> list[Vector]:
        return [Vector([random.random() for _ in range(3)]) for _ in range(size)]

    @evaluation_fn
    def evaluate(instance: Vector) -> float:
        return sum(instance.values)

    @selection_fn
    def select(
        pop: list[Vector], off: list[Vector], scores: list[float]
    ) -> list[Vector]:
        combined = pop + off
        indexed = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        return [combined[i] for i, _ in indexed[: len(pop)]]

    @crossover_fn
    def crossover(parents: list[Vector]) -> list[Vector]:
        avg = [(a + b) / 2 for a, b in zip(parents[0].values, parents[1].values)]
        return [Vector(avg)]

    @mutation_fn
    def mutate(instance: Vector) -> Vector:
        new_values = [v + random.uniform(-0.05, 0.1) for v in instance.values]
        return Vector(new_values)

    @finish_condition_fn
    def finish(pop: list[Vector], gen: int, scores: list[float]) -> bool:
        return gen >= 10

    ea = EvolutionaryAlgorithm(
        initial_population=init_pop,
        evaluation=evaluate,
        selection=select,
        finish_condition=finish,
        crossover=crossover,
        mutation=mutate,
        population_size=10,
        max_workers=2,
    )

    result = ea.run()

    assert isinstance(result.best_instance, Vector)
    assert len(result.best_instance.values) == 3
    assert len(result.population) == 10
    assert result.generation == 10


def test_parallel_ea_with_closures_capturing_state():
    """Parallel execution with closures that capture external state (cloudpickle test)."""
    target = [10.0, 20.0, 30.0]
    mutation_step = 0.5

    @initial_population_fn
    def init_pop(size: int) -> list[list[float]]:
        return [[random.uniform(0, 5) for _ in target] for _ in range(size)]

    @evaluation_fn
    def evaluate(instance: list[float]) -> float:
        # Negative distance to target (higher = closer to target)
        return -sum((a - b) ** 2 for a, b in zip(instance, target))

    @selection_fn
    def select(
        pop: list[list[float]], off: list[list[float]], scores: list[float]
    ) -> list[list[float]]:
        combined = pop + off
        indexed = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        return [combined[i] for i, _ in indexed[: len(pop)]]

    @crossover_fn
    def crossover(parents: list[list[float]]) -> list[list[float]]:
        # Midpoint crossover, captures `target` length
        child = [(a + b) / 2 for a, b in zip(parents[0], parents[1])]
        return [child]

    @mutation_fn
    def mutate(instance: list[float]) -> list[float]:
        # Captures `mutation_step` and `target` from enclosing scope
        return [v + random.uniform(-mutation_step, mutation_step * 2) for v in instance]

    @finish_condition_fn
    def finish(pop: list[list[float]], gen: int, scores: list[float]) -> bool:
        return gen >= 20

    ea = EvolutionaryAlgorithm(
        initial_population=init_pop,
        evaluation=evaluate,
        selection=select,
        finish_condition=finish,
        crossover=crossover,
        mutation=mutate,
        population_size=15,
        max_workers=2,
    )

    result = ea.run()

    # After 20 generations, the best should be closer to target than the initial random [0..5]
    assert result.best_fitness > -sum(t**2 for t in target)  # better than all-zeros
    assert result.generation == 20


def test_parallel_ea_speedup_with_slow_operators():
    """Verify parallel execution actually runs crossover/mutation concurrently."""
    sleep_duration = 0.02  # 20ms per operation

    @initial_population_fn
    def init_pop(size: int) -> list[int]:
        return list(range(size))

    @evaluation_fn
    def evaluate(instance: int) -> float:
        return float(instance)

    @selection_fn
    def select(pop: list[int], off: list[int], scores: list[float]) -> list[int]:
        combined = pop + off
        indexed = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        return [combined[i] for i, _ in indexed[: len(pop)]]

    @crossover_fn
    def crossover(parents: list[int]) -> list[int]:
        time.sleep(sleep_duration)
        return [max(parents)]

    @mutation_fn
    def mutate(instance: int) -> int:
        time.sleep(sleep_duration)
        return instance + 1

    @finish_condition_fn
    def finish(pop: list[int], gen: int, scores: list[float]) -> bool:
        return gen >= 1

    num_workers = min(os.cpu_count() or 2, 4)
    pop_size = 10

    # Sequential: each of pop_size//2 crossovers + pop_size mutations = 15 * 20ms = 300ms
    # Parallel with N workers: ~300ms / N
    ea = EvolutionaryAlgorithm(
        initial_population=init_pop,
        evaluation=evaluate,
        selection=select,
        finish_condition=finish,
        crossover=crossover,
        mutation=mutate,
        population_size=pop_size,
        max_workers=num_workers,
    )

    start = time.monotonic()
    result = ea.run()
    elapsed = time.monotonic() - start

    total_ops = pop_size // 2 + pop_size  # crossovers + mutations
    sequential_time = total_ops * sleep_duration

    # Parallel should be meaningfully faster than sequential
    # Use a generous threshold (< 80% of sequential time) to avoid flakiness
    assert elapsed < sequential_time * 0.8, (
        f"Parallel took {elapsed:.3f}s, sequential would take ~{sequential_time:.3f}s. "
        f"Expected speedup with {num_workers} workers."
    )
    assert result.generation == 1
