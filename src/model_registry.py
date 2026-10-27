"""Local Ollama model options."""


CANDIDATE_MODELS = [
    {
        "id": "qwen3-0.6b",
        "family": "Qwen",
        "ollama": "qwen3:0.6b",
        "params_b": 0.6,
        "licence": "Apache-2.0",
        "resource_tier": "tiny",
        "expected_memory_gb_q4": 0.7,
        "why": "Smallest option and suitable for low-memory CPUs.",
    },
    {
        "id": "qwen3-1.7b",
        "family": "Qwen",
        "ollama": "qwen3:1.7b",
        "params_b": 1.7,
        "licence": "Apache-2.0",
        "resource_tier": "small",
        "expected_memory_gb_q4": 1.5,
        "why": "Stronger than 0.6B while still practical on a laptop CPU.",
    },
    {
        "id": "qwen2.5-1.5b-instruct",
        "family": "Qwen",
        "ollama": "qwen2.5:1.5b",
        "params_b": 1.5,
        "licence": "Apache-2.0",
        "resource_tier": "small",
        "expected_memory_gb_q4": 1.4,
        "why": "Fallback option when Qwen3 is unavailable.",
    },
    {
        "id": "llama3.2-1b-instruct",
        "family": "Llama",
        "ollama": "llama3.2:1b",
        "params_b": 1.0,
        "licence": "Meta Llama Community License",
        "resource_tier": "small",
        "expected_memory_gb_q4": 1.0,
        "why": "Small CPU model with a more restrictive licence.",
    },
    {
        "id": "llama3.2-3b-instruct",
        "family": "Llama",
        "ollama": "llama3.2:3b",
        "params_b": 3.0,
        "licence": "Meta Llama Community License",
        "resource_tier": "medium",
        "expected_memory_gb_q4": 2.4,
        "why": "Larger option for systems with more memory.",
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
    """Pick the first preferred installed model."""
    # Follow the fixed preference order.
    installed = set(installed_names)
    for name in MODEL_PREFERENCE:
        if name in installed:
            return name
    return sorted(installed)[0] if installed else None


def model_profile_by_ollama(name):
    """Find a model profile by Ollama name."""
    # Keep custom models usable without registry entries.
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
