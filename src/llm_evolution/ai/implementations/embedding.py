from typing import Any
from openai import OpenAI
from llm_evolution.ai.interfaces.embedding import EmbeddingModel


class OpenAIEmbedding(EmbeddingModel):
    """
    Implementation of the EmbeddingModel interface using OpenAI-compatible endpoints.

    This implementation works with OpenAI's official API as well as any
    compatible service by configuring the base_url.
    """

    def __init__(
        self,
        model: str,
        api_key: str = "sk-no-key-required",
        base_url: str | None = None,
        **kwargs: Any,
    ):
        """
        Initialize the OpenAI Embedding model.

        Args:
            model: The name of the embedding model to use.
            api_key: The API key for authentication.
            base_url: The base URL of the API endpoint.
                     Allows using any OpenAI-compatible provider.
            **kwargs: Additional arguments passed to the OpenAI client.
        """
        self.model = model
        self.client = OpenAI(api_key=api_key, base_url=base_url, **kwargs)

    def __call__(self, text: str) -> list[float]:
        """
        Generates an embedding vector using the OpenAI embeddings endpoint.

        Args:
            text: The input text to embed.

        Returns:
            list[float]: The embedding vector.
        """
        response = self.client.embeddings.create(
            model=self.model,
            input=text,
        )
        return response.data[0].embedding
