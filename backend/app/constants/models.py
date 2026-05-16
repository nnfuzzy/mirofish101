"""Allow-list of LLM models exposed to the per-simulation picker.

Verify the exact LiteLLM identifiers against current LiteLLM Gemini docs
before merging — preview model names sometimes carry a version suffix
(e.g. -preview-001). Update SUPPORTED_MODELS here in one place; both the
backend validator and the frontend dropdown read from it.
"""

SUPPORTED_MODELS = [
    {
        "id": "gemini/gemini-3.1-pro-preview",
        "display_name": "Pro Preview",
        "tagline": "Best quality, slowest",
        "cost_tier": "$$$",
    },
    {
        "id": "gemini/gemini-3-flash-preview",
        "display_name": "Flash Preview",
        "tagline": "Balanced (default)",
        "cost_tier": "$$",
    },
    {
        "id": "gemini/gemini-3.1-flash-lite",
        "display_name": "Flash Lite",
        "tagline": "Fastest, cheapest",
        "cost_tier": "$",
    },
]

DEFAULT_MODEL_ID = "gemini/gemini-3-flash-preview"
SUPPORTED_MODEL_IDS = {m["id"] for m in SUPPORTED_MODELS}
