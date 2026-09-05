from openai import OpenAI
from dotenv import load_dotenv

from src.rag.rags import RAGBase
from src.rag.prompt import SYSTEM_PROMPT

load_dotenv()


def run_graphrag(query: str):
    client = OpenAI()
    rag_system = RAGBase(client, instructions=SYSTEM_PROMPT, model="gpt-5.6-luna")
    response = rag_system.rag(query=query)
    return response


if __name__ == "__main__":
    query = "what do you know about AirAsia Zest?"
    response = run_graphrag(query)
    print(response)
