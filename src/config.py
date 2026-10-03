from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv
from model_provider import ProviderConfig, normalize_provider


@dataclass
class LabConfig:
    """Shared configuration for the lab."""

    base_dir: Path
    data_dir: Path
    state_dir: Path
    compact_threshold_tokens: int
    compact_keep_messages: int
    model: ProviderConfig
    judge_model: ProviderConfig


def load_config(base_dir: Path | None = None) -> LabConfig:
    """Load environment variables and return a populated LabConfig."""
    root = (base_dir or Path(__file__).resolve().parent.parent).resolve()

    # Load environment variables if .env exists
    env_file = root / ".env"
    if env_file.exists():
        load_dotenv(env_file)

    # Ensure state directory exists
    state_dir = root / "state"
    state_dir.mkdir(parents=True, exist_ok=True)
    (state_dir / "profiles").mkdir(parents=True, exist_ok=True)

    data_dir = root / "data"

    # Compact memory tuning
    compact_threshold = int(os.getenv("COMPACT_THRESHOLD_TOKENS", "500"))
    compact_keep = int(os.getenv("COMPACT_KEEP_MESSAGES", "4"))

    # Provider configuration for primary model
    raw_provider = os.getenv("LLM_PROVIDER", "openai")
    provider = normalize_provider(raw_provider)
    model_name = os.getenv("LLM_MODEL", "gpt-4o-mini")

    api_key = (
        os.getenv("OPENAI_API_KEY")
        or os.getenv("GEMINI_API_KEY")
        or os.getenv("ANTHROPIC_API_KEY")
        or os.getenv("OPENROUTER_API_KEY")
        or os.getenv("CUSTOM_API_KEY")
    )
    base_url = os.getenv("CUSTOM_BASE_URL") or os.getenv("OLLAMA_BASE_URL")

    model_config = ProviderConfig(
        provider=provider,
        model_name=model_name,
        temperature=0.0,
        api_key=api_key,
        base_url=base_url,
    )

    # Provider configuration for judge model
    raw_judge_provider = os.getenv("JUDGE_PROVIDER", provider)
    judge_provider = normalize_provider(raw_judge_provider)
    judge_model_name = os.getenv("JUDGE_MODEL", model_name)
    judge_api_key = os.getenv("JUDGE_API_KEY", api_key)
    judge_base_url = os.getenv("JUDGE_BASE_URL", base_url)

    judge_config = ProviderConfig(
        provider=judge_provider,
        model_name=judge_model_name,
        temperature=0.0,
        api_key=judge_api_key,
        base_url=judge_base_url,
    )

    return LabConfig(
        base_dir=root,
        data_dir=data_dir,
        state_dir=state_dir,
        compact_threshold_tokens=compact_threshold,
        compact_keep_messages=compact_keep,
        model=model_config,
        judge_model=judge_config,
    )
