

from dotenv import load_dotenv
from openai import OpenAI
import yaml

from src.utils.llm_schemas import OfflineJudgeResult
from src.evaluation.judge_prompt import build_messages

from src.utils.llm_schemas import OfflineJudgeResult, OnlineJudgeResult


with open("configs/config.yaml", "r") as f:
    config = yaml.safe_load(f)

load_dotenv()

judge_client = OpenAI()

def judge_offline_qa(question: str, 
                     generated_answer: str, 
                     ground_truth_answer: str, 
                     gold_chunk: str) -> OfflineJudgeResult:
    """Evaluates synthetic QA pairs."""
    
    response = judge_client.beta.chat.completions.parse(
        model=config["judgement"]["model"],
        response_format=OfflineJudgeResult,
        messages=build_messages(query=question,
                                gold_chunk=gold_chunk,
                                ground_truth_answer=ground_truth_answer,
                                generated_answer=generated_answer),
    )
    return response.choices[0].message.parsed



def judge_online_question(question: str, generated_answer: str) -> OnlineJudgeResult:
    """Evaluates user questions in production."""

    system_prompt = """
You are an expert evaluator for a conversational AI system.
Your task is to evaluate if the 'Generated Answer' is relevant to the 'Question'."""

    prompt = f"""
    Question: {question}
    Generated Answer: {generated_answer}
    
    Rate the relevancy: RELEVANT, PARTIALLY_RELEVANT, or NON_RELEVANT.
    """
    response = judge_client.beta.chat.completions.parse(
        model=config["judgement"]["model"],
        response_format=OnlineJudgeResult,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ],
    )
    return response.choices[0].message.parsed