import random
from dataclasses import dataclass, field

from llm_evolution.ai.interfaces.llm import LLM, Message
from llm_evolution.ai.interfaces.embedding import EmbeddingModel
from llm_evolution.ai.interfaces.vector_db import VectorDatabase
from llm_evolution.interfaces.mutation import Mutation
from llm_evolution.interfaces.crossover import Crossover


@dataclass
class ActionableThought:
    """A thought consisting of natural language description and code examples."""

    description: str
    code_examples: list[str]
    id: str = field(default_factory=lambda: str(random.randint(0, 1000000)))


class EvolutionOfKernels:
    """
    Implementation of 'Evolution of Kernels' for mutation and crossover.
    Reference: https://arxiv.org/abs/2509.14265
    """

    def __init__(
        self,
        llm: LLM,
        embedding_model: EmbeddingModel,
        vector_db: VectorDatabase,
        mutation_probability: float = 0.1,
        n_thoughts: int = 3,
    ):
        self.llm = llm
        self.embedding_model = embedding_model
        self.vector_db = vector_db
        self.mutation_probability = mutation_probability
        self.n_thoughts = n_thoughts

    def get_mutation(self) -> Mutation[str]:
        """Returns a Mutation protocol implementation."""
        return self._MutationWrapper(self)

    def get_crossover(self) -> Crossover[str]:
        """Returns a Crossover protocol implementation."""
        return self._CrossoverWrapper(self)

    def ingest_thought(self, thought: ActionableThought) -> None:
        """
        Ingest an actionable thought into the vector database.

        Args:
            thought: The actionable thought to ingest.
        """
        embedding = self.embedding_model(thought.description)
        self.vector_db.add(
            ids=[thought.id],
            embeddings=[embedding],
            metadatas=[
                {
                    "description": thought.description,
                    "code_examples": "|||".join(thought.code_examples),
                }
            ],
            documents=[thought.description],
        )

    class _MutationWrapper(Mutation[str]):
        def __init__(self, parent: "EvolutionOfKernels"):
            self.parent = parent

        def __call__(self, instance: str) -> str | None:
            if random.random() > self.parent.mutation_probability:
                return None

            # 1. Compute embedding of the program (instance)
            program_embedding = self.parent.embedding_model(instance)

            # 2. Search in vector database for top N thoughts
            results = self.parent.vector_db.query(
                query_embeddings=[program_embedding], n_results=self.parent.n_thoughts
            )

            thoughts_context = ""
            for i, res in enumerate(results):
                desc = res["metadata"].get("description", "")
                examples = res["metadata"].get("code_examples", "").split("|||")
                thoughts_context += (
                    f"Thought {i + 1}: {desc}\nExamples:\n"
                    + "\n".join(examples)
                    + "\n\n"
                )

            # 3. Prompt LLM to mutate the program using the thoughts
            prompt = [
                Message(
                    role="system",
                    content="You are an expert in kernel optimization. Use the provided optimization thoughts to improve the given program.",
                ),
                Message(
                    role="user",
                    content=f"Original Program:\n{instance}\n\nOptimization Thoughts:\n{thoughts_context}\n\nPlease provide the mutated and optimized version of the program. Return ONLY the code.",
                ),
            ]

            mutated_program = self.parent.llm(prompt)
            return mutated_program.strip()

    class _CrossoverWrapper(Crossover[str]):
        def __init__(self, parent: "EvolutionOfKernels"):
            self.parent = parent

        def __call__(self, parents: list[str]) -> list[str]:
            if not parents:
                return []

            parents_context = "\n\n".join(
                [f"Parent {i + 1}:\n{p}" for i, p in enumerate(parents)]
            )

            prompt = [
                Message(
                    role="system",
                    content="You are an expert in kernel optimization. Combine the best features of the following parent programs into a single optimized offspring program.",
                ),
                Message(
                    role="user",
                    content=f"{parents_context}\n\nPlease provide the combined and optimized version of the program. Return ONLY the code.",
                ),
            ]

            offspring = self.parent.llm(prompt)
            return [offspring.strip()]
