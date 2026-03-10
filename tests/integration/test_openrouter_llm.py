import os

import pytest

from llm_evolution.ai.implementations.llm import open_router_model
from llm_evolution.ai.interfaces.llm import Message
from dotenv import load_dotenv

load_dotenv()


@pytest.mark.skipif(
    not os.getenv("OPENROUTER_API_KEY"),
    reason="OPENROUTER_API_KEY not set; skipping OpenRouter integration test.",
)
def test_openrouter_llm_route_works():
    llm = open_router_model("openai/gpt-oss-20b")

    response = llm(
        [
            Message(role="system", content="You are a helpful assistant."),
            Message(role="user", content="Reply with exactly: OK"),
        ]
    )

    assert isinstance(response, str)
    assert response.strip() != ""
