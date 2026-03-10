import pytest

from llm_evolution.ai.interfaces.react_llm import (
    ReActLLM,
    retry_message_builder_fn,
    termination_condition_fn,
)
from llm_evolution.ai.interfaces.llm import Message


class RecordingLLM:
    def __init__(self, responses: list[str]) -> None:
        self._responses = responses
        self.calls: list[list[Message]] = []
        self._index = 0

    def __call__(self, messages: list[Message]) -> str:
        self.calls.append(list(messages))
        response = self._responses[self._index]
        self._index += 1
        return response


def test_react_llm_retries_until_condition_met():
    llm = RecordingLLM(["draft", "better", "final"])

    class AcceptsFinal:
        def evaluate(self, messages: list[Message], response: str) -> str | None:
            if response == "final":
                return None
            return "Response must be 'final'."

    react_llm = ReActLLM(llm, AcceptsFinal(), max_iterations=5)

    response = react_llm([Message(role="user", content="Start")])

    assert response == "final"
    assert len(llm.calls) == 3
    assert llm.calls[0][0].content == "Start"
    assert llm.calls[1][-1].role == "user"
    assert "Response must be 'final'." in llm.calls[1][-1].content


def test_react_llm_uses_custom_retry_message_builder():
    llm = RecordingLLM(["first", "second"])

    @termination_condition_fn
    def accepts_second(messages: list[Message], response: str) -> str | None:
        if response == "second":
            return None
        return "Need the second response."

    @retry_message_builder_fn
    def retry_message_builder(
        response: str, feedback: str, messages: list[Message]
    ) -> Message:
        return Message(role="system", content=f"Retry after: {response} ({feedback})")

    react_llm = ReActLLM(
        llm,
        accepts_second,
        max_iterations=3,
        retry_message_builder=retry_message_builder,
    )

    react_llm([Message(role="user", content="Start")])

    assert llm.calls[1][-1].role == "system"
    assert llm.calls[1][-1].content == "Retry after: first (Need the second response.)"


def test_react_llm_raises_after_max_iterations():
    llm = RecordingLLM(["one", "two"])

    class NeverAccepts:
        def evaluate(self, messages: list[Message], response: str) -> str | None:
            return "Still not acceptable."

    react_llm = ReActLLM(llm, NeverAccepts(), max_iterations=2)

    with pytest.raises(RuntimeError):
        react_llm([Message(role="user", content="Start")])
