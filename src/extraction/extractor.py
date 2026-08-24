
from typing import Any, Generic
import json
from pydantic import ValidationError

from src.utils.structures import Usage, ExtractionResult, ExtractionStatus
from src.utils.llm_schemas import T
from src.utils.providers import LLMProvider, OpenAIProvider, OllamaProvider
from src.utils.token_utils import calculate_openai_usage
    

# ---------------------------------------------------------------------------
# Information extractor
# ---------------------------------------------------------------------------

class InformationExtractor(Generic[T]):
    """
    It returns strcutred outputs aiming:
    - to extract structured information from documents. 
    - make decision about retrieval strategy (graph or vector)
    """

    def __init__(
        self,
        provider: LLMProvider,
        input_template: str,
        schema: type[T],
        params: dict
    ):
        self.provider = provider
        self.model = params['model']
        self.prompt_version = params['instruction_version']
        self.instruction = params['instruction']
        self.prompt_template = input_template
        self.schema = schema
        self.run_id = params['run_id']
        self.run_name = params['run_name']

    def build_prompt(self, text: str) -> str:
        return self.prompt_template.format(text=text).strip()

    def extract(self, 
                text: str, 
                document_id: str | None = None) -> ExtractionResult[T]:

        prompt = self.build_prompt(text)

        messages = [
            {
                "role": "system",
                "content": self.instruction,
            },
            {
                "role": "user",
                "content": prompt,
            },
        ]
        
        response, raw_output = self.provider.generate(
            model=self.model,
            messages=messages,
            schema=self.schema,
        )

        
        usage = self.calculate_usage(response)
        
        # ---------------------------------------------------------------
        # OpenAI structured output
        # ---------------------------------------------------------------

        if isinstance(self.provider, OpenAIProvider):

            output = response.choices[0].message.parsed

            if output is not None:
                return ExtractionResult(
                    data=output,
                    raw_output=None,
                    status=ExtractionStatus.SUCCESS,
                    error=None,
                    usage=usage,
                    model=self.model,
                    run_id=self.run_id,
                    document_id=document_id,                    
                )

            return ExtractionResult(
                data=None,
                raw_output=response.choices[0].message.content,
                status=ExtractionStatus.SCHEMA_MISMATCH,
                error="OpenAI returned no parsed structured output.",
                usage=usage,
                model=self.model,
                run_id=self.run_id,
                document_id=document_id,
            )

        # ---------------------------------------------------------------
        # Ollama / raw JSON output
        # ---------------------------------------------------------------

        if raw_output is None or not raw_output.strip():
            return ExtractionResult(
                data=None,
                raw_output=raw_output,
                status=ExtractionStatus.EMPTY_OUTPUT,
                error="Model returned empty output.",
                usage=usage,
                model=self.model,
                run_id=self.run_id,
                document_id=document_id,
            )

        try:
            data = json.loads(raw_output)

        except json.JSONDecodeError as exc:
            return ExtractionResult(
                data=None,
                raw_output=raw_output,
                status=ExtractionStatus.INVALID_JSON,
                error=str(exc),
                usage=usage,
                model=self.model,
                run_id=self.run_id,
                document_id=document_id,
            )

        return self.validate_data(
            data=data,
            raw_output=raw_output,
            usage=usage,
            document_id=document_id,
        )

        

    def validate_data(
        self,
        data: Any,
        raw_output: str | None,
        usage: Usage,
        document_id: str | None = None,
    ) -> ExtractionResult[T]:

        try:
            output = self.schema.model_validate(data)

        except ValidationError as exc:
            return ExtractionResult(
                data=None,
                raw_output=raw_output,
                status=ExtractionStatus.SCHEMA_MISMATCH,
                error=str(exc),
                usage=usage,
                model=self.model,
                run_id=self.run_id,
                document_id=document_id,
            )

        return ExtractionResult(
            data=output,
            raw_output=raw_output,
            status=ExtractionStatus.SUCCESS,
            error=None,
            usage=usage,
            model=self.model,
            run_id=self.run_id,
            document_id=document_id,
        )
    

    def calculate_usage(self, response: Any) -> Usage:

        if isinstance(self.provider, OpenAIProvider):
            return self._calculate_openai_usage(response)

        if isinstance(self.provider, OllamaProvider):
            return self._calculate_ollama_usage(response)

        return Usage()

    def _calculate_openai_usage(self, response: Any) -> Usage:        
        return calculate_openai_usage(self.model, response)


    def _calculate_ollama_usage(self, response: Any) -> Usage:

        return Usage(
            input_tokens=response.prompt_eval_count or 0,
            output_tokens=response.eval_count or 0,
            cached_tokens=0,
            cost=0.0,
        )
