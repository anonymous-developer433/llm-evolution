from collections.abc import Callable
from typing import Protocol

from llm_evolution.ai.interfaces.llm import LLM, Message


class TerminationCondition(Protocol):
    """Defines when a response is acceptable or needs feedback-driven retrying."""

    def evaluate(self, messages: list[Message], response: str) -> str | None:
        """
        Return feedback when the response is insufficient, or None when acceptable.

        Args:
            messages: The current conversation history including the latest response.
            response: The most recent assistant response.

        Returns:
            str | None: Feedback for a retry, or None if the response is accepted.
        """
        ...


def termination_condition_fn(fn: Callable[[list[Message], str], str | None]):
    """Decorator to convert a function into a TerminationCondition implementation."""

    class Wrapper:
        def __init__(self, func: Callable[[list[Message], str], str | None]) -> None:
            self.func = func

        def evaluate(self, messages: list[Message], response: str) -> str | None:
            return self.func(messages, response)

    return Wrapper(fn)


class RetryMessageBuilder(Protocol):
    """Builds the next retry message using feedback from the termination condition."""

    def build(self, response: str, feedback: str, messages: list[Message]) -> Message:
        """
        Create the next message to append after feedback.

        Args:
            response: The most recent assistant response.
            feedback: Feedback explaining why a retry is needed.
            messages: The current conversation history including the latest response.

        Returns:
            Message: The message to append for the retry.
        """
        ...


def retry_message_builder_fn(fn: Callable[[str, str, list[Message]], Message]):
    """Decorator to convert a function into a RetryMessageBuilder implementation."""

    class Wrapper:
        def __init__(self, func: Callable[[str, str, list[Message]], Message]) -> None:
            self.func = func

        def build(
            self, response: str, feedback: str, messages: list[Message]
        ) -> Message:
            return self.func(response, feedback, messages)

    return Wrapper(fn)


@retry_message_builder_fn
def _default_retry_message(
    response: str, feedback: str, _messages: list[Message]
) -> Message:
    return Message(
        role="user",
        content=(
            "The previous response did not satisfy the termination condition. "
            "Feedback: "
            f"{feedback}\n"
            "Please try again, improve it, and keep any useful parts of the prior attempt."
        ),
    )


class ReActLLM(LLM):
    """LLM wrapper that retries until a termination condition is satisfied."""

    def __init__(
        self,
        llm: LLM,
        termination_condition: TerminationCondition,
        max_iterations: int = 5,
        retry_message_builder: RetryMessageBuilder | None = None,
    ) -> None:
        if max_iterations < 1:
            raise ValueError("max_iterations must be at least 1")

        self._llm = llm
        self._termination_condition = termination_condition
        self._max_iterations = max_iterations
        self._retry_message_builder = retry_message_builder or _default_retry_message

    def __call__(self, messages: list[Message]) -> str:
        conversation = list(messages)

        for _ in range(self._max_iterations):
            response = self._llm(conversation)
            conversation.append(Message(role="assistant", content=response))

            feedback = self._termination_condition.evaluate(conversation, response)
            if not feedback:
                return response

            conversation.append(
                self._retry_message_builder.build(response, feedback, conversation)
            )

        raise RuntimeError(
            "ReActLLM exceeded max_iterations without satisfying the termination condition."
        )
