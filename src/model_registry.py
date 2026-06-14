"""
Candidate local model registry for Variant 3.

This is a research/runtime registry, not a downloader. The runtime can use
matching Ollama model names when they are already installed locally.
"""


CANDIDATE_MODELS = [
    {
        "id": "qwen3-0.6b",
        "family": "Qwen",
        "ollama": "qwen3:0.6b",
        "params_b": 0.6,
        "licence": "Apache-2.0",
        "resource_tier": "tiny",
        "expected_memory_gb_q4": 0.7,
        "why": "Best first choice for no-lag CPU RAG; permissive licence and very small memory footprint.",
    },
    {
        "id": "qwen3-1.7b",
        "family": "Qwen",
        "ollama": "qwen3:1.7b",
        "params_b": 1.7,
        "licence": "Apache-2.0",
        "resource_tier": "small",
        "expected_memory_gb_q4": 1.5,
        "why": "Better answer quality than 0.6B while still practical on commodity CPU.",
    },
    {
        "id": "qwen2.5-1.5b-instruct",
        "family": "Qwen",
        "ollama": "qwen2.5:1.5b",
        "params_b": 1.5,
        "licence": "Apache-2.0",
        "resource_tier": "small",
        "expected_memory_gb_q4": 1.4,
        "why": "Stable instruct model; useful fallback when Qwen3 is unavailable.",
    },
    {
        "id": "llama3.2-1b-instruct",
        "family": "Llama",
        "ollama": "llama3.2:1b",
        "params_b": 1.0,
        "licence": "Meta Llama Community License",
        "resource_tier": "small",
        "expected_memory_gb_q4": 1.0,
        "why": "Good CPU option, but licence is more restrictive than Apache-2.0 alternatives.",
    },
    {
        "id": "llama3.2-3b-instruct",
        "family": "Llama",
        "ollama": "llama3.2:3b",
        "params_b": 3.0,
        "licence": "Meta Llama Community License",
        "resource_tier": "medium",
        "expected_memory_gb_q4": 2.4,
        "why": "Higher quality tier when latency target and memory budget allow it.",
    },
]


MODEL_PREFERENCE = [
    "qwen3:0.6b",
    "qwen3:1.7b",
    "qwen2.5:1.5b",
    "llama3.2:1b",
    "llama3.2:3b",
]


def preferred_model(installed_names):
    """Pick the preferred installed Ollama model name."""
    installed = set(installed_names)
    for name in MODEL_PREFERENCE:
        if name in installed:
            return name
    return sorted(installed)[0] if installed else None


def model_profile_by_ollama(name):
    """Return registry metadata for an Ollama model name."""
    for profile in CANDIDATE_MODELS:
        if profile["ollama"] == name:
            return profile
    return {
        "id": name or "none",
        "family": "unknown",
        "ollama": name,
        "params_b": None,
        "licence": "unknown",
        "resource_tier": "custom",
        "expected_memory_gb_q4": None,
        "why": "User-provided local model.",
    }
