"""Call a local Ollama model."""

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass

from .model_registry import model_profile_by_ollama, preferred_model
from .prompts import estimate_tokens


DEFAULT_ENDPOINT = "http://127.0.0.1:11434"
DEFAULT_SEED = 42


@dataclass
class GenerationResult:
    text: str
    ok: bool
    backend: str
    model: str
    latency_ms: float
    prompt_tokens: int
    output_tokens: int
    error: str = ""
    raw: dict = None


class LocalLLM:
    """Small wrapper around Ollama's local API."""

    def __init__(self, model=None, endpoint=None, seed=DEFAULT_SEED):
        # Allow local settings through environment variables.
        self.endpoint = (endpoint or os.environ.get("LOCAL_QA_OLLAMA_ENDPOINT")
                         or DEFAULT_ENDPOINT).rstrip("/")
        self.model = model or os.environ.get("LOCAL_QA_MODEL")
        self.seed = seed

    def _request_json(self, method, path, payload=None, timeout_s=0.25):
        # Encode request bodies only when present.
        data = None
        headers = {}
        if payload is not None:
            data = json.dumps(payload).encode("utf-8")
            headers["Content-Type"] = "application/json"

        req = urllib.request.Request(
            self.endpoint + path,
            data=data,
            headers=headers,
            method=method,
        )
        # Read the full non-streaming response.
        with urllib.request.urlopen(req, timeout=timeout_s) as response:
            return json.loads(response.read().decode("utf-8"))

    def available_models(self, timeout_s=0.25):
        """List installed models."""
        # Treat an offline server as no available models.
        try:
            payload = self._request_json("GET", "/api/tags", timeout_s=timeout_s)
        except (OSError, urllib.error.URLError, TimeoutError, json.JSONDecodeError):
            return []
        return [item.get("name", "") for item in payload.get("models", [])
                if item.get("name")]

    def status(self):
        """Report local model status."""
        # Prefer an explicit model over automatic selection.
        models = self.available_models()
        selected = self.model or preferred_model(models)
        return {
            "backend": "ollama",
            "endpoint": self.endpoint,
            "available": bool(models),
            "installed_models": models,
            "selected_model": selected,
            "profile": model_profile_by_ollama(selected) if selected else None,
        }

    def generate(self, prompt, max_new_tokens=160, timeout_s=1.5,
                 temperature=0.0, top_p=1.0, num_ctx=1024):
        """Generate text or return a fallback error."""
        # Check the server before sending the prompt.
        t0 = time.perf_counter()
        models = self.available_models(timeout_s=min(timeout_s, 0.25))
        model = self.model or preferred_model(models)
        prompt_tokens = estimate_tokens(prompt)

        if not model:
            # Return structured errors so callers can fall back.
            return GenerationResult(
                text="",
                ok=False,
                backend="ollama",
                model="",
                latency_ms=(time.perf_counter() - t0) * 1000,
                prompt_tokens=prompt_tokens,
                output_tokens=0,
                error="no_local_model_available",
                raw={},
            )

        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "think": False,
            "options": {
                # Disable sampling for stable output.
                "seed": self.seed,
                "temperature": temperature,
                "top_p": top_p,
                "num_predict": max_new_tokens,
                "num_ctx": num_ctx,
            },
        }

        try:
            raw = self._request_json(
                "POST", "/api/chat", payload=payload, timeout_s=timeout_s
            )
        except TimeoutError:
            # Keep timeouts distinct in audit logs.
            return GenerationResult(
                text="",
                ok=False,
                backend="ollama",
                model=model,
                latency_ms=(time.perf_counter() - t0) * 1000,
                prompt_tokens=prompt_tokens,
                output_tokens=0,
                error="generation_timeout",
                raw={},
            )
        except (OSError, urllib.error.URLError, json.JSONDecodeError) as exc:
            return GenerationResult(
                text="",
                ok=False,
                backend="ollama",
                model=model,
                latency_ms=(time.perf_counter() - t0) * 1000,
                prompt_tokens=prompt_tokens,
                output_tokens=0,
                error=str(exc) or "generation_failed",
                raw={},
            )

        # Ollama returns generated text inside message.content.
        text = raw.get("message", {}).get("content", "").strip()
        return GenerationResult(
            text=text,
            ok=bool(text),
            backend="ollama",
            model=model,
            latency_ms=(time.perf_counter() - t0) * 1000,
            prompt_tokens=raw.get("prompt_eval_count", prompt_tokens),
            output_tokens=raw.get("eval_count", estimate_tokens(text)),
            error="" if text else "empty_generation",
            raw=raw,
        )
