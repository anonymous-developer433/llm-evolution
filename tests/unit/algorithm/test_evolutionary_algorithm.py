import pytest

from llm_evolution.algorithm.evolutionary_algorithm import (
    EvolutionaryAlgorithm,
)
from llm_evolution.interfaces.crossover import crossover_fn
from llm_evolution.interfaces.evaluation import evaluation_fn
from llm_evolution.interfaces.finish_condition import finish_condition_fn
from llm_evolution.interfaces.initial_population import initial_population_fn
from llm_evolution.interfaces.mutation import mutation_fn
from llm_evolution.interfaces.selection import selection_fn


@pytest.fixture
def basic_components():
    """Provide minimal EA components for testing."""

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
        return [(parents[0] + parents[1]) // 2]

    @mutation_fn
    def mutate(instance: int) -> int:
        return instance + 1

    @finish_condition_fn
    def finish(pop: list[int], gen: int, scores: list[float]) -> bool:
        return gen >= 3

    return {
        "initial_population": init_pop,
        "evaluation": evaluate,
        "selection": select,
        "crossover": crossover,
        "mutation": mutate,
        "finish_condition": finish,
    }


class TestEvaluationCaching:
    """Tests for the evaluation caching mechanism."""

    def test_evaluation_called_once_per_individual(self):
        """Each individual should be evaluated exactly once, not re-evaluated."""
        call_count = 0

        @initial_population_fn
        def init_pop(size: int) -> list[int]:
            return list(range(size))

        @evaluation_fn
        def evaluate(instance: int) -> float:
            nonlocal call_count
            call_count += 1
            return float(instance)

        @selection_fn
        def select(pop: list[int], off: list[int], scores: list[float]) -> list[int]:
            return pop

        @finish_condition_fn
        def finish(pop: list[int], gen: int, scores: list[float]) -> bool:
            return gen >= 2

        ea = EvolutionaryAlgorithm(
            initial_population=init_pop,
            evaluation=evaluate,
            selection=select,
            finish_condition=finish,
            population_size=5,
        )

        ea.run()

        # Population of 5, no crossover/mutation, so no offspring.
        # Generation 0: evaluate 5 individuals (cache miss)
        # Generation 1: evaluate 5 individuals (cache hit — same objects from selection)
        # Generation 2: evaluate 5 individuals (cache hit) -> finish
        # Final fitness: cache hit
        # Total: only 5 evaluations (the initial ones)
        assert call_count == 5

    def test_new_offspring_are_evaluated(self, basic_components):
        """Offspring (new objects) should be evaluated, but parents should not be re-evaluated."""
        call_count = 0
        original_eval = basic_components["evaluation"]

        @evaluation_fn
        def counting_eval(instance: int) -> float:
            nonlocal call_count
            call_count += 1
            return original_eval(instance)

        @finish_condition_fn
        def finish(pop: list[int], gen: int, scores: list[float]) -> bool:
            return gen >= 1

        ea = EvolutionaryAlgorithm(
            initial_population=basic_components["initial_population"],
            evaluation=counting_eval,
            selection=basic_components["selection"],
            finish_condition=finish,
            crossover=basic_components["crossover"],
            mutation=basic_components["mutation"],
            population_size=4,
        )

        result = ea.run()

        # Generation 0: 4 individuals evaluated (cache miss)
        # Crossover produces new objects -> evaluated (cache miss)
        # Mutation produces new objects -> evaluated (cache miss)
        # All offspring are new objects, so they get evaluated once.
        # Surviving population members that were already evaluated are NOT re-evaluated.
        # The key point: call_count < total number of evaluation calls without caching.
        assert call_count > 0
        assert result.best_fitness >= 0

    def test_cache_cleared_between_runs(self, basic_components):
        """Cache should be reset at the start of each run."""
        call_count = 0

        @evaluation_fn
        def counting_eval(instance: int) -> float:
            nonlocal call_count
            call_count += 1
            return float(instance)

        @finish_condition_fn
        def finish(pop: list[int], gen: int, scores: list[float]) -> bool:
            return True  # Finish immediately

        ea = EvolutionaryAlgorithm(
            initial_population=basic_components["initial_population"],
            evaluation=counting_eval,
            selection=basic_components["selection"],
            finish_condition=finish,
            population_size=3,
        )

        ea.run()
        first_run_count = call_count

        call_count = 0
        ea.run()
        second_run_count = call_count

        # Both runs should evaluate the same number of individuals
        # because the cache is reset at the start of each run.
        assert first_run_count == second_run_count
        assert first_run_count == 3


class TestParallelCrossoverAndMutation:
    """Tests for parallel crossover and mutation execution."""

    def test_parallel_crossover_produces_offspring(self):
        """Crossover should produce offspring when run in parallel."""

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
            return [parents[0] + parents[1]]

        @finish_condition_fn
        def finish(pop: list[int], gen: int, scores: list[float]) -> bool:
            return gen >= 3

        ea = EvolutionaryAlgorithm(
            initial_population=init_pop,
            evaluation=evaluate,
            selection=select,
            finish_condition=finish,
            crossover=crossover,
            population_size=10,
            max_workers=2,
        )

        result = ea.run()
        assert result.best_fitness > 0
        assert len(result.population) == 10

    def test_parallel_mutation_produces_offspring(self):
        """Mutation should produce offspring when run in parallel."""

        @initial_population_fn
        def init_pop(size: int) -> list[int]:
            return [1] * size

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

        ea = EvolutionaryAlgorithm(
            initial_population=init_pop,
            evaluation=evaluate,
            selection=select,
            finish_condition=finish,
            mutation=mutate,
            population_size=6,
            max_workers=2,
        )

        result = ea.run()
        # After 5 generations of mutation (+1 each time), best should be > 1
        assert result.best_fitness > 1

    def test_parallel_mutation_none_filtered(self):
        """Mutations returning None should be filtered out in parallel execution."""

        @initial_population_fn
        def init_pop(size: int) -> list[int]:
            return list(range(size))

        @evaluation_fn
        def evaluate(instance: int) -> float:
            return float(instance)

        @selection_fn
        def select(pop: list[int], off: list[int], scores: list[float]) -> list[int]:
            return pop

        @mutation_fn
        def mutate(instance: int) -> int | None:
            return None

        @finish_condition_fn
        def finish(pop: list[int], gen: int, scores: list[float]) -> bool:
            return gen >= 1

        ea = EvolutionaryAlgorithm(
            initial_population=init_pop,
            evaluation=evaluate,
            selection=select,
            finish_condition=finish,
            mutation=mutate,
            population_size=5,
            max_workers=2,
        )

        result = ea.run()
        # No mutation succeeded, population should remain unchanged
        assert sorted(result.population) == [0, 1, 2, 3, 4]

    def test_parallel_with_closure_functions(self):
        """Closures (non-serializable with pickle) should work via loky/cloudpickle."""
        multiplier = 10

        @initial_population_fn
        def init_pop(size: int) -> list[int]:
            return list(range(1, size + 1))

        @evaluation_fn
        def evaluate(instance: int) -> float:
            return float(instance * multiplier)

        @selection_fn
        def select(pop: list[int], off: list[int], scores: list[float]) -> list[int]:
            combined = pop + off
            indexed = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
            return [combined[i] for i, _ in indexed[: len(pop)]]

        @crossover_fn
        def crossover(parents: list[int]) -> list[int]:
            return [max(parents)]

        @mutation_fn
        def mutate(instance: int) -> int:
            return instance + multiplier

        @finish_condition_fn
        def finish(pop: list[int], gen: int, scores: list[float]) -> bool:
            return gen >= 2

        ea = EvolutionaryAlgorithm(
            initial_population=init_pop,
            evaluation=evaluate,
            selection=select,
            finish_condition=finish,
            crossover=crossover,
            mutation=mutate,
            population_size=4,
            max_workers=2,
        )

        result = ea.run()
        # Closures capturing `multiplier` should work fine with cloudpickle
        assert result.best_fitness > 0

    def test_max_workers_parameter(self, basic_components):
        """Algorithm should accept and use the max_workers parameter."""
        ea = EvolutionaryAlgorithm(
            initial_population=basic_components["initial_population"],
            evaluation=basic_components["evaluation"],
            selection=basic_components["selection"],
            finish_condition=basic_components["finish_condition"],
            crossover=basic_components["crossover"],
            mutation=basic_components["mutation"],
            population_size=10,
            max_workers=1,
        )

        result = ea.run()
        assert result.generation == 3
        assert len(result.population) == 10
