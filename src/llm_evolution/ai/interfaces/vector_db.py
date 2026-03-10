from typing import Protocol, Any, runtime_checkable


@runtime_checkable
class VectorDatabase(Protocol):
    """Protocol for vector databases."""

    def add(
        self,
        ids: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict[str, Any]],
        documents: list[str],
    ) -> None:
        """
        Add items to the vector database.

        Args:
            ids: Unique identifiers for the items.
            embeddings: Embedding vectors for the items.
            metadatas: Metadata for each item.
            documents: Original documents or descriptions.
        """
        ...

    def query(
        self, query_embeddings: list[list[float]], n_results: int = 10
    ) -> list[dict[str, Any]]:
        """
        Query the vector database for similar items.

        Args:
            query_embeddings: Embedding vectors to search for.
            n_results: Number of results to return.

        Returns:
            list[dict[str, Any]]: List of query results.
        """
        ...
