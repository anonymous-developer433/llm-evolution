from dataclasses import dataclass
from typing import Literal, Protocol, runtime_checkable


@dataclass
class Message:
    role: Literal["user", "assistant", "system", "tool"]
    content: str


@runtime_checkable
class LLM(Protocol):
    """Protocol for Large Language Models."""

    def __call__(self, messages: list[Message]) -> str:
        """
        Generates a text response from a list of messages.

        Args:
            messages: A list of message objects.

        Returns:
            str: The generated text response.
        """
        ...


def llm_fn(fn):
    """
    Decorator to convert a function into an LLM protocol implementation.

    Args:
        fn: A function that takes a list of messages and returns a text response.

    Returns:
        Wrapper: A class implementing the LLM protocol.
    """

    class Wrapper:
        def __init__(self, func):
            self.func = func

        def __call__(self, messages: list[Message]) -> str:
            return self.func(messages)

    return Wrapper(fn)
