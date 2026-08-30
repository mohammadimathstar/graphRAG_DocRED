from .structures import Usage
from typing import Any


MODEL_PRICING = {
    "gpt-5.4-nano": {
        "input_price": 0.20 / 1_000_000,
        "cached_price": 0.02 / 1_000_000,
        "output_price": 1.25 / 1_000_000,
    },
    "gpt-5.4-mini": {
        "input_price": 0.75 / 1_000_000,
        "cached_price": 0.075 / 1_000_000,
        "output_price": 4.50 / 1_000_000,
    },
    "gpt-5.6-luna": {
        "input_price": 0.20 / 1_000_000,
        "cached_price": 0.02 / 1_000_000,
        "output_price": 1.20 / 1_000_000,
    },
}


def calculate_openai_usage(model, response: Any) -> Usage:
    pricing = MODEL_PRICING.get(model)
    usage = response.usage

    input_tokens = getattr(usage, "prompt_tokens", None) or getattr(
        usage, "input_tokens", None
    )
    output_tokens = getattr(usage, "completion_tokens", None) or getattr(
        usage, "output_tokens", None
    )

    cached_tokens = 0
    if getattr(usage, "prompt_tokens_details", None):
        cached_tokens = usage.prompt_tokens_details.cached_tokens or 0
    elif getattr(usage, "input_tokens_details", None):
        cached_tokens = usage.input_tokens_details.cached_tokens

    if pricing is None:
        return Usage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cached_tokens=cached_tokens,
            cost=None,
        )

    billable_input = max(0, input_tokens - cached_tokens)

    cost = (
        billable_input * pricing["input_price"]
        + cached_tokens * pricing["cached_price"]
        + output_tokens * pricing["output_price"]
    )

    return Usage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cached_tokens=cached_tokens,
        cost=cost,
    )
