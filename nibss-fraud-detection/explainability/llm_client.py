"""
Week 4 — LLM client wrapper.

Provider-agnostic single-function interface (explain_prompt) so the
underlying model can be swapped (e.g. Claude) without touching callers in
generate_explanations.py. Default provider is Gemini, per design.md.

Gemini's free tier caps gemini-2.5-flash at 20 requests/day (discovered
empirically, not just a per-minute limit as first assumed — see
design.md's Sample Size amendment). To keep the pipeline usable on a free
key, this client falls back through a chain of models on a per-day quota
exhaustion (RESOURCE_EXHAUSTED with a "PerDay" quotaId) rather than
retry-waiting, since a per-day quota won't reset within a reasonable retry
window. Per-minute rate limits still retry-with-backoff on the same model.

Requires GEMINI_API_KEY (or GOOGLE_API_KEY) to be set in the environment.
"""
import os
import re
import time
from pathlib import Path

# Tried in order; each entry falls back to the next on a per-day quota
# exhaustion for that model specifically.
MODEL_FALLBACK_CHAIN = [
    "gemini-2.5-flash",
    "gemini-2.0-flash",
    "gemini-2.5-flash-lite",
]
DEFAULT_MODEL_NAME = MODEL_FALLBACK_CHAIN[0]

MIN_SECONDS_BETWEEN_CALLS = 13.0

_ENV_PATH = Path(__file__).resolve().parent / ".env"


def _load_dotenv(path: Path = _ENV_PATH) -> None:
    """Minimal .env loader (avoids adding python-dotenv as a dependency).
    Does not override variables already set in the environment."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip()
        os.environ.setdefault(key, value)


_load_dotenv()


def _should_skip_to_next_model(exc: Exception) -> bool:
    text = str(exc)
    return ("RESOURCE_EXHAUSTED" in text and "PerDay" in text) or "NOT_FOUND" in text


def _retry_delay_seconds(exc: Exception) -> float | None:
    match = re.search(r"retryDelay['\"]?\s*:\s*['\"]?(\d+)s", str(exc))
    return float(match.group(1)) if match else None


class LLMClient:
    """Falls back through MODEL_FALLBACK_CHAIN on daily quota exhaustion.
    self.model_name always reflects whichever model actually served the
    most recent successful call."""

    def __init__(
        self,
        model_chain: list[str] | None = None,
        api_key: str | None = None,
    ):
        from google import genai

        api_key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError(
                "No Gemini API key found. Set GEMINI_API_KEY (or GOOGLE_API_KEY) "
                "in the environment before running generate_explanations.py."
            )
        self._client = genai.Client(api_key=api_key)
        self._model_chain = list(model_chain or MODEL_FALLBACK_CHAIN)
        self._exhausted_models: set[str] = set()
        self._last_call_time_by_model: dict[str, float] = {}
        self.model_name = self._model_chain[0]

    def _throttle(self, model: str):
        last = self._last_call_time_by_model.get(model)
        if last is not None:
            remaining = MIN_SECONDS_BETWEEN_CALLS - (time.time() - last)
            if remaining > 0:
                time.sleep(remaining)

    def _active_models(self):
        return [m for m in self._model_chain if m not in self._exhausted_models]

    def explain_prompt(self, prompt: str, max_retries_per_model: int = 3) -> str:
        last_error = None
        for model in self._active_models():
            for attempt in range(max_retries_per_model):
                self._throttle(model)
                try:
                    response = self._client.models.generate_content(
                        model=model, contents=prompt
                    )
                    self._last_call_time_by_model[model] = time.time()
                    self.model_name = model
                    return response.text.strip()
                except Exception as exc:
                    last_error = exc
                    self._last_call_time_by_model[model] = time.time()
                    if _should_skip_to_next_model(exc):
                        self._exhausted_models.add(model)
                        break  # move to next model in the chain immediately
                    wait = _retry_delay_seconds(exc) or (2 ** attempt) + 5
                    time.sleep(wait)
        raise RuntimeError(
            f"LLM call failed on all models in {self._model_chain}: {last_error}"
        )
