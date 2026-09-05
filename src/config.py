"""Application settings loaded from environment variables / .env file."""
from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    openai_api_key: str
    chat_model: str = "gpt-4o-mini"


def load_settings() -> Settings:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY environment variable is not set (check your .env file).")
    chat_model = os.getenv("CHAT_MODEL", "gpt-4o-mini")
    return Settings(openai_api_key=api_key, chat_model=chat_model)
