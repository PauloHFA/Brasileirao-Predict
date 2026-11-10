import time
from typing import Any, Dict

import httpx

from src.config import settings


def check_llm_provider_health() -> Dict[str, Any]:
    """
    Checks the connectivity and status of the currently configured LLM provider.
    Returns status: 'online' | 'degraded' | 'offline' | 'mock_active'.
    """
    provider = settings.LLM_PROVIDER
    start_time = time.time()

    if provider == "mock":
        return {
            "provider": "mock",
            "model": "rule-anchored-mock-v1",
            "status": "mock_active",
            "latency_ms": round((time.time() - start_time) * 1000, 2),
            "message": "Deterministic mock provider active. Zero external dependencies."
        }

    if provider == "ollama":
        base_url = settings.LLM_BASE_URL
        try:
            resp = httpx.get(f"{base_url}/api/tags", timeout=3.0)
            latency = round((time.time() - start_time) * 1000, 2)
            if resp.status_code == 200:
                data = resp.json()
                models = [m.get("name") for m in data.get("models", [])]
                configured_model = settings.LLM_MODEL
                is_installed = any(configured_model in m for m in models)
                return {
                    "provider": "ollama",
                    "base_url": base_url,
                    "model": configured_model,
                    "model_installed": is_installed,
                    "available_models": models,
                    "status": "online" if is_installed else "degraded",
                    "latency_ms": latency,
                    "message": "Ollama server connected." if is_installed else f"Model '{configured_model}' not found in Ollama. Available: {models}"
                }
            return {
                "provider": "ollama",
                "status": "offline",
                "latency_ms": latency,
                "message": f"Ollama returned HTTP {resp.status_code}"
            }
        except Exception as e:
            return {
                "provider": "ollama",
                "status": "offline",
                "latency_ms": round((time.time() - start_time) * 1000, 2),
                "message": f"Could not connect to Ollama at {base_url}: {str(e)}"
            }

    if provider == "openai":
        if not settings.OPENAI_API_KEY:
            return {
                "provider": "openai",
                "status": "not_configured",
                "message": "OPENAI_API_KEY is not set."
            }
        return {
            "provider": "openai",
            "model": settings.LLM_MODEL or "gpt-4o-mini",
            "status": "online",
            "message": "OpenAI API configured."
        }

    if provider == "gemini":
        if not settings.GEMINI_API_KEY:
            return {
                "provider": "gemini",
                "status": "not_configured",
                "message": "GEMINI_API_KEY is not set."
            }
        return {
            "provider": "gemini",
            "model": settings.LLM_MODEL or "gemini-1.5-flash",
            "status": "online",
            "message": "Gemini API configured."
        }

    return {
        "provider": provider,
        "status": "unknown",
        "message": f"Unknown LLM provider: {provider}"
    }
