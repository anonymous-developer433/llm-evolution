from llm_evolution.ai.interfaces.llm import llm_fn, LLM, Message
from llm_evolution.ai.interfaces.embedding import embedding_model_fn, EmbeddingModel


def test_llm_decorator():
    @llm_fn
    def my_llm(messages: list[Message]) -> str:
        return "response"

    assert isinstance(my_llm, LLM)
    assert my_llm([Message(role="user", content="hi")]) == "response"


def test_embedding_decorator():
    @embedding_model_fn
    def my_embedding(text: str) -> list[float]:
        return [0.1, 0.2]

    assert isinstance(my_embedding, EmbeddingModel)
    assert my_embedding("hi") == [0.1, 0.2]
