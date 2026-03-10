import pytest
from unittest.mock import Mock

from llm_evolution.implementations.evolution_of_kernels import (
    EvolutionOfKernels,
    ActionableThought,
)
from llm_evolution.ai.interfaces.llm import LLM
from llm_evolution.ai.interfaces.embedding import EmbeddingModel
from llm_evolution.ai.interfaces.vector_db import VectorDatabase
from llm_evolution.interfaces.mutation import Mutation
from llm_evolution.interfaces.crossover import Crossover


@pytest.fixture
def mock_llm():
    llm = Mock(spec=LLM)
    llm.return_value = "Optimized Code"
    return llm


@pytest.fixture
def mock_embedding_model():
    model = Mock(spec=EmbeddingModel)
    model.return_value = [0.1, 0.2, 0.3]
    return model


@pytest.fixture
def mock_vector_db():
    db = Mock(spec=VectorDatabase)
    db.query.return_value = [
        {
            "id": "1",
            "metadata": {"description": "T1", "code_examples": "ex1|||ex2"},
            "document": "T1",
            "distance": 0.1,
        }
    ]
    return db


@pytest.fixture
def evolution_of_kernels(mock_llm, mock_embedding_model, mock_vector_db):
    return EvolutionOfKernels(
        llm=mock_llm,
        embedding_model=mock_embedding_model,
        vector_db=mock_vector_db,
        mutation_probability=1.0,  # Always mutate for tests
        n_thoughts=1,
    )


def test_ingest_thought(evolution_of_kernels, mock_embedding_model, mock_vector_db):
    thought = ActionableThought(
        description="Test thought", code_examples=["code1", "code2"]
    )
    evolution_of_kernels.ingest_thought(thought)

    mock_embedding_model.assert_called_once_with("Test thought")
    mock_vector_db.add.assert_called_once()

    # Check that code_examples were joined
    args, kwargs = mock_vector_db.add.call_args
    assert "code1|||code2" in kwargs["metadatas"][0]["code_examples"]


def test_mutation_logic(
    evolution_of_kernels, mock_llm, mock_embedding_model, mock_vector_db
):
    program = "original_program"
    mutation_op = evolution_of_kernels.get_mutation()
    mutated = mutation_op(program)

    assert mutated == "Optimized Code"
    mock_embedding_model.assert_called_with(program)
    mock_vector_db.query.assert_called_once()
    mock_llm.assert_called_once()

    # Verify prompt content
    prompt = mock_llm.call_args[0][0]
    assert "original_program" in prompt[1].content
    assert "T1" in prompt[1].content
    assert "ex1" in prompt[1].content


def test_crossover_logic(evolution_of_kernels, mock_llm):
    parents = ["parent1", "parent2"]
    crossover_op = evolution_of_kernels.get_crossover()
    offspring = crossover_op(parents)

    assert len(offspring) == 1
    assert offspring[0] == "Optimized Code"
    mock_llm.assert_called_once()

    # Verify prompt content
    prompt = mock_llm.call_args[0][0]
    assert "parent1" in prompt[1].content
    assert "parent2" in prompt[1].content


def test_protocols_implementation(evolution_of_kernels):
    # Check if they implement the protocols
    assert isinstance(evolution_of_kernels.get_mutation(), Mutation)
    assert isinstance(evolution_of_kernels.get_crossover(), Crossover)
