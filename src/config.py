from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(BASE_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore"
    )

    DATABASE_URL: str = f"sqlite:///{BASE_DIR / 'data' / 'brpredictor.db'}"
    LLM_PROVIDER: Literal["mock", "ollama", "openai", "gemini"] = "mock"
    LLM_BASE_URL: str = "http://localhost:11434"
    LLM_MODEL: str = "llama3.2"
    OPENAI_API_KEY: str = ""
    GEMINI_API_KEY: str = ""
    NEWS_STORAGE_POLICY: str = "METADATA_ONLY"
    AS_OF_HOURS: int = 24
    MODEL_REGISTRY_PATH: str = str(BASE_DIR / "artifacts")
    LOG_LEVEL: str = "INFO"
    ADMIN_API_TOKEN: str = "dev-secret-token-123"

settings = Settings()
