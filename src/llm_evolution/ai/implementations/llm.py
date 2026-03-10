import os
from typing import Any, cast

from openai import OpenAI
from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam

from llm_evolution.ai.interfaces.llm import LLM, Message


class OpenAILLM(LLM):
    """
    Implementation of the LLM interface using OpenAI-compatible endpoints.

    This implementation works with OpenAI's official API as well as any
    compatible service (e.g., LocalAI, vLLM, Ollama) by configuring the base_url.
    """

    def __init__(
        self,
        model: str,
        api_key: str = "sk-no-key-required",
        base_url: str | None = None,
        **kwargs: Any,
    ):
        """
        Initialize the OpenAI LLM.

        Args:
            model: The name of the model to use.
            api_key: The API key for authentication.
            base_url: The base URL of the API endpoint.
                     Allows using any OpenAI-compatible provider.
            **kwargs: Additional arguments passed to the OpenAI client.
        """
        self.model = model
        self.client = OpenAI(api_key=api_key, base_url=base_url, **kwargs)

    def __call__(self, messages: list[Message]) -> str:
        """
        Generates a text response using the OpenAI chat completion endpoint.

        Args:
            messages: A list of message objects.

        Returns:
            str: The generated response text.
        """
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages_to_openai(messages),
        )
        return response.choices[0].message.content or ""


def messages_to_openai(messages: list[Message]) -> list[ChatCompletionMessageParam]:
    """
    Converts a list of Message objects to a list of OpenAI chat completion message parameters.

    Args:
        messages: A list of message objects.

    Returns:
        list[ChatCompletionMessageParam]: A list of OpenAI chat completion message parameters.
    """
    return [
        cast(
            ChatCompletionMessageParam,
            {"role": message.role, "content": message.content},
        )
        for message in messages
    ]


def open_router_model(model: str) -> LLM:
    """
    Returns an OpenAI LLM instance with the specified model for the OpenRouter API.

    Args:
        model: The name of the model to use.

    Returns:
        LLM: An OpenAI LLM instance with the specified model for the OpenRouter API.
    """
    from dotenv import load_dotenv

    load_dotenv()

    api_key = os.getenv("OPENROUTER_API_KEY")

    if not api_key:
        raise ValueError("OPENROUTER_API_KEY not found in environment variables")

    api_url = "https://openrouter.ai/api/v1"
    return OpenAILLM(model, api_key=api_key, base_url=api_url)
