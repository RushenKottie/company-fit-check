"""Tests for LLM client factories and adapters."""

from __future__ import annotations

from types import SimpleNamespace

from langchain_core.messages import HumanMessage, SystemMessage

from llm.client import AnthropicFoundryChatAdapter, create_user_simulator_chat_model


def test_anthropic_foundry_adapter_translates_messages(monkeypatch):
    captured: dict[str, object] = {}

    class FakeMessagesApi:
        def create(self, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(content=[SimpleNamespace(text="Paris")])

    class FakeAnthropicFoundry:
        def __init__(self, *, api_key: str, base_url: str) -> None:
            captured["api_key"] = api_key
            captured["base_url"] = base_url
            self.messages = FakeMessagesApi()

    monkeypatch.setattr(
        "llm.client.importlib.import_module",
        lambda name: SimpleNamespace(AnthropicFoundry=FakeAnthropicFoundry),
    )

    adapter = AnthropicFoundryChatAdapter(
        api_key="key",
        base_url="https://example.test/anthropic",
        model="claude-haiku-4-5",
        max_tokens=123,
    )
    response = adapter.invoke(
        [
            SystemMessage(content="system prompt"),
            HumanMessage(content="What is the capital of France?"),
        ]
    )

    assert captured == {
        "api_key": "key",
        "base_url": "https://example.test/anthropic",
        "model": "claude-haiku-4-5",
        "system": "system prompt",
        "messages": [{"role": "user", "content": "What is the capital of France?"}],
        "max_tokens": 123,
    }
    assert response.content[0].text == "Paris"


def test_user_simulator_chat_model_prefers_foundry_when_configured(monkeypatch):
    create_user_simulator_chat_model.cache_clear()
    monkeypatch.setattr(
        "llm.client.importlib.import_module",
        lambda name: SimpleNamespace(
            AnthropicFoundry=lambda **kwargs: SimpleNamespace(messages=SimpleNamespace(create=lambda **create_kwargs: None))
        ),
    )
    monkeypatch.setattr(
        "llm.client.get_user_simulator_foundry_settings",
        lambda: SimpleNamespace(
            is_configured=True,
            api_key="key",
            endpoint="https://example.test/anthropic",
            model="claude-haiku-4-5",
            max_tokens=321,
        ),
    )
    monkeypatch.setattr(
        "llm.client.create_user_simulator_azure_chat_model",
        lambda: "azure-model",
    )

    model = create_user_simulator_chat_model()

    assert isinstance(model, AnthropicFoundryChatAdapter)
    create_user_simulator_chat_model.cache_clear()


def test_user_simulator_chat_model_returns_none_when_foundry_not_configured(monkeypatch):
    create_user_simulator_chat_model.cache_clear()
    monkeypatch.setattr(
        "llm.client.get_user_simulator_foundry_settings",
        lambda: SimpleNamespace(
            is_configured=False,
            api_key=None,
            endpoint=None,
            model=None,
            max_tokens=321,
        ),
    )
    monkeypatch.setattr(
        "llm.client.create_user_simulator_azure_chat_model",
        lambda: "azure-model",
    )

    model = create_user_simulator_chat_model()

    assert model is None
    create_user_simulator_chat_model.cache_clear()
