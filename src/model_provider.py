from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass
class ProviderConfig:
    """Provider configuration shared by the agents.

    Required providers for this lab:
    - openai
    - custom (OpenAI-compatible base URL)
    - gemini
    - anthropic
    - ollama
    - openrouter
    """

    provider: str
    model_name: str
    temperature: float = 0.0
    api_key: str | None = None
    base_url: str | None = None


def normalize_provider(value: str) -> str:
    """Map provider names and common aliases to canonical provider names."""
    if not value:
        return "openai"
    val = value.strip().lower()
    mapping = {
        "openai": "openai",
        "custom": "custom",
        "gemini": "gemini",
        "google": "gemini",
        "google-genai": "gemini",
        "google_genai": "gemini",
        "anthropic": "anthropic",
        "anthorpic": "anthropic",
        "claude": "anthropic",
        "ollama": "ollama",
        "openrouter": "openrouter",
        "open-router": "openrouter",
        "open_router": "openrouter",
    }
    if val in mapping:
        return mapping[val]
    raise ValueError(f"Unsupported provider: {value}")


def build_chat_model(config: ProviderConfig):
    """Instantiate the real chat model for the selected provider.

    Mapping:
    - `openai` -> `ChatOpenAI`
    - `custom` -> `ChatOpenAI` with `base_url`
    - `gemini` -> `ChatGoogleGenerativeAI`
    - `anthropic` -> `ChatAnthropic`
    - `ollama` -> `ChatOllama`
    - `openrouter` -> `ChatOpenRouter`
    """
    provider = normalize_provider(config.provider)

    if provider == "openai":
        from langchain_openai import ChatOpenAI

        api_key = config.api_key or os.getenv("OPENAI_API_KEY") or "dummy-key"
        return ChatOpenAI(
            model=config.model_name,
            temperature=config.temperature,
            api_key=api_key,
        )

    if provider == "custom":
        from langchain_openai import ChatOpenAI

        api_key = config.api_key or os.getenv("CUSTOM_API_KEY") or "dummy-key"
        base_url = config.base_url or os.getenv("CUSTOM_BASE_URL", "http://localhost:8000/v1")
        return ChatOpenAI(
            model=config.model_name,
            temperature=config.temperature,
            api_key=api_key,
            base_url=base_url,
        )

    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        api_key = config.api_key or os.getenv("GEMINI_API_KEY") or "dummy-key"
        return ChatGoogleGenerativeAI(
            model=config.model_name,
            temperature=config.temperature,
            google_api_key=api_key,
        )

    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        api_key = config.api_key or os.getenv("ANTHROPIC_API_KEY") or "dummy-key"
        return ChatAnthropic(
            model=config.model_name,
            temperature=config.temperature,
            api_key=api_key,
        )

    if provider == "ollama":
        from langchain_ollama import ChatOllama

        base_url = config.base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        return ChatOllama(
            model=config.model_name,
            temperature=config.temperature,
            base_url=base_url,
        )

    if provider == "openrouter":
        from langchain_openrouter import ChatOpenRouter

        api_key = config.api_key or os.getenv("OPENROUTER_API_KEY") or "dummy-key"
        return ChatOpenRouter(
            model=config.model_name,
            temperature=config.temperature,
            api_key=api_key,
        )

    raise ValueError(f"Unhandled provider: {provider}")
