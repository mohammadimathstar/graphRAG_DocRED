from abc import ABC, abstractmethod
from typing import Any

import ollama
from openai import OpenAI
from pydantic import BaseModel


# ---------------------------------------------------------------------------
# Provider abstraction
# ---------------------------------------------------------------------------


class LLMProvider(ABC):
    @abstractmethod
    def generate(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        schema: type[BaseModel],
    ) -> tuple[Any, str | None]:
        """
        Generate an LLM response.

        Returns:
            response: Native provider response.
            raw_output: Textual model output, if available.
        """
        pass


# ---------------------------------------------------------------------------
# OpenAI provider
# ---------------------------------------------------------------------------


class OpenAIProvider(LLMProvider):
    def __init__(self, client: OpenAI):
        self.client = client

    def generate(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        schema: type[BaseModel],
    ) -> tuple[Any, str | None]:
        response = self.client.beta.chat.completions.parse(
            model=model,
            messages=messages,
            response_format=schema,
        )

        return response, None


# ---------------------------------------------------------------------------
# Ollama provider
# ---------------------------------------------------------------------------


class OllamaProvider(LLMProvider):
    def __init__(self, client: ollama.Client):
        self.client = client

    def generate(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        schema: type[BaseModel],
    ) -> tuple[Any, str | None]:
        response = self.client.chat(
            model=model,
            messages=messages,
            format=schema.model_json_schema(),
            options={
                "temperature": 0,
            },
        )

        raw_output = (
            response["message"]["content"]
            if isinstance(response, dict)
            else response.message.content
        )

        return response, raw_output
