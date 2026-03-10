from unittest.mock import Mock

from llm_evolution.algorithm.evolutionary_algorithm import EvolutionaryAlgorithm
from llm_evolution.implementations.evolution_of_kernels import EvolutionOfKernels
from llm_evolution.ai.interfaces.llm import LLM
from llm_evolution.ai.interfaces.embedding import EmbeddingModel
from llm_evolution.ai.interfaces.vector_db import VectorDatabase
from llm_evolution.interfaces.mutation import Mutation
from llm_evolution.interfaces.crossover import Crossover
from llm_evolution.interfaces.initial_population import InitialPopulation
from llm_evolution.interfaces.evaluation import Evaluation
from llm_evolution.interfaces.selection import Selection
from llm_evolution.interfaces.finish_condition import FinishCondition


def test_ea_injection_with_evolution_of_kernels():
    # 1. Setup mocks for AI dependencies
    mock_llm = Mock(spec=LLM)
    mock_llm.return_value = "Optimized Code"

    mock_embedding = Mock(spec=EmbeddingModel)
    mock_embedding.return_value = [0.1, 0.2]

    mock_db = Mock(spec=VectorDatabase)
    mock_db.query.return_value = []

    # 2. Create EvolutionOfKernels instance
    eok = EvolutionOfKernels(
        llm=mock_llm,
        embedding_model=mock_embedding,
        vector_db=mock_db,
        mutation_probability=1.0,
        n_thoughts=1,
    )

    # 3. Setup mocks for EA strategy dependencies
    mock_init_pop = Mock(spec=InitialPopulation)
    mock_init_pop.return_value = ["code1", "code2"]

    mock_eval = Mock(spec=Evaluation)
    mock_eval.return_value = 1.0

    mock_selection = Mock(spec=Selection)
    mock_selection.return_value = ["code1", "code2"]

    mock_finish = Mock(spec=FinishCondition)
    mock_finish.side_effect = [False, True]  # Run for 1 generation

    # 4. Initialize EA with EvolutionOfKernels wrappers
    ea = EvolutionaryAlgorithm[str](
        initial_population=mock_init_pop,
        evaluation=mock_eval,
        selection=mock_selection,
        finish_condition=mock_finish,
        mutation=eok.get_mutation(),
        crossover=eok.get_crossover(),
        population_size=2,
    )

    # 5. Run EA
    result = ea.run()

    # 6. Verify that mutation and crossover were called
    assert result.generation == 1
    mock_llm.assert_called()  # Should be called by mutation or crossover

    # Ensure wrappers correctly implement protocols
    assert isinstance(ea.mutation, Mutation)
    assert isinstance(ea.crossover, Crossover)
