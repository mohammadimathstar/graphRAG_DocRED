from src.utils.structures import Usage
from typing import Any
import time
import yaml
import os

from src.retrieval.retrieval import retrieve_context
from src.extraction.extractor import OpenAIProvider, InformationExtractor
from src.utils.llm_schemas import RouterDecision
from src.retrieval.prompt import ROUTER_PROMPT
from src.utils.token_utils import calculate_openai_usage

from src.db.connection import get_conn_from_pool, release_conn
from src.db.ingest import log_production_trace

USER_PROMPT_TEMPLATE = """
Question: 
{question}

Context:
{context}
"""


# Get the directory where this script (rags.py) lives
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
# Go up two levels (from src/rag/ to the project root)
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
# Build the absolute path to the config file
CONFIG_PATH = os.path.join(PROJECT_ROOT, "configs", "config.yaml")

with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)


class RAGBase:
    def __init__(self, llm_client, instructions, model="gpt-5.4-nano"):
        self.llm_client = llm_client
        self.instructions = instructions
        self.model = model
        self.provider = OpenAIProvider(llm_client)

        self._init_router()

    def _init_router(self):
        params_to_track = {
            "model": self.model,
            "run_id": None,
            "run_name": None,
            "instruction_version": config["retrieval"]["instruction_version"],
            "instruction": ROUTER_PROMPT,
        }

        input_template = """
        {text}
        """

        self.router = InformationExtractor(
            provider=self.provider,
            schema=RouterDecision,
            input_template=input_template,
            params=params_to_track,
        )

    def search(self, question: str, num_results: int = 3):
        """Checks out a connection from the pool, runs retrieval, and returns it."""
        conn = get_conn_from_pool()
        try:
            return retrieve_context(conn, self.router, question, num_results)
        finally:
            # ALWAYS return the connection to the pool
            release_conn(conn)

    def build_context(self, search_results: list):
        return "\n\n".join(search_results)

    def build_prompt(self, question: str, search_results: list[str]) -> str:
        context = self.build_context(search_results)
        prompt = USER_PROMPT_TEMPLATE.format(question=question, context=context)
        return prompt.strip()

    def _calculate_openai_usage(self, response: Any) -> Usage:
        return calculate_openai_usage(self.model, response)

    def llm(self, prompt):
        message_history = [
            {"role": "system", "content": self.instructions},
            {"role": "user", "content": prompt},
        ]

        response = self.llm_client.responses.create(
            model=self.model, input=message_history
        )

        return response

    def rag(self, query: str, session_id: str = None):
        start_time = time.time()

        retrieved_contexts = []
        retrieval_method = "unknown"
        retrieval_usage = Usage()

        # 1. Retrieval
        try:
            print("Retrieval step...")
            retrieval_result = self.search(query)

            retrieved_contexts = retrieval_result.get("context", [])
            retrieval_method = retrieval_result.get("strategy", "unknown")
            retrieval_usage = retrieval_result.get("usage", Usage())

        except Exception as e:
            # Log failure and exit early
            latency_ms = int((time.time() - start_time) * 1000)
            trace_id = log_production_trace(
                user_question=query,
                retrieved_context=[],
                retrieval_method="failed",
                retrieval_usage=Usage(),
                generated_answer="",
                generator_model=self.model,
                generator_usage=Usage(),
                latency_ms=latency_ms,
                status="failed_retrieval",
                error_message=str(e),
                session_id=session_id,
            )

            return "Sorry, I encountered an error finding information."

        # 2. Generation
        try:
            print("Building prompt...")
            prompt = self.build_prompt(query, retrieved_contexts)

            print("Generating answer...")
            response = self.llm(prompt)

            answer = response.output_text
            generation_usage = self._calculate_openai_usage(response)
        except Exception as e:
            latency_ms = int((time.time() - start_time) * 1000)
            trace_id = log_production_trace(
                user_question=query,
                retrieved_context=retrieved_contexts,
                retrieval_method=retrieval_method,
                retrieval_usage=retrieval_usage,
                generated_answer="",
                generator_model=self.model,
                generator_usage=Usage(),
                latency_ms=latency_ms,
                status="failed_generation",
                error_message=str(e),
                session_id=session_id,
            )
            return "Sorry, I encountered an error generating the answer."

        # 3. Log Success
        latency_ms = int((time.time() - start_time) * 1000)

        trace_id = log_production_trace(
            user_question=query,
            retrieved_context=retrieved_contexts,
            retrieval_method=retrieval_method,
            retrieval_usage=retrieval_usage,
            generated_answer=answer,
            generator_model=self.model,
            generator_usage=generation_usage,
            latency_ms=latency_ms,
            status="success",
            session_id=session_id,
        )

        return {
            "answer": answer, 
            "trace_id": trace_id,
        }
