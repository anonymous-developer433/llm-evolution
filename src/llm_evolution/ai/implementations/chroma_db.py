from typing import Any, cast
import chromadb

from llm_evolution.ai.interfaces.vector_db import VectorDatabase


class ChromaDBImplementation(VectorDatabase):
    """Implementation of VectorDatabase using ChromaDB."""

    def __init__(
        self,
        collection_name: str = "evolution_of_kernels",
        persist_directory: str | None = None,
    ):
        """
        Initialize ChromaDB implementation.

        Args:
            collection_name: Name of the collection to use.
            persist_directory: Directory to persist data. If None, uses in-memory.
        """
        if persist_directory:
            self.client = chromadb.PersistentClient(path=persist_directory)
        else:
            self.client = chromadb.EphemeralClient()

        self.collection = self.client.get_or_create_collection(name=collection_name)

    def add(
        self,
        ids: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict[str, Any]],
        documents: list[str],
    ) -> None:
        """Add items to the ChromaDB collection."""
        self.collection.add(
            ids=ids,
            embeddings=cast(Any, embeddings),
            metadatas=cast(Any, metadatas),
            documents=documents,
        )

    def query(
        self, query_embeddings: list[list[float]], n_results: int = 10
    ) -> list[dict[str, Any]]:
        """Query the ChromaDB collection for similar items."""
        results = self.collection.query(
            query_embeddings=cast(Any, query_embeddings), n_results=n_results
        )

        # Flatten and format results to match expected interface output
        formatted_results = []
        if results["ids"]:
            for i in range(len(results["ids"][0])):
                formatted_results.append(
                    {
                        "id": results["ids"][0][i],
                        "metadata": results["metadatas"][0][i]
                        if results["metadatas"]
                        else {},
                        "document": results["documents"][0][i]
                        if results["documents"]
                        else "",
                        "distance": results["distances"][0][i]
                        if results["distances"]
                        else 0.0,
                    }
                )

        return formatted_results
