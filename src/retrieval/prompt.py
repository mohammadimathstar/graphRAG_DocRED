ROUTER_PROMPT = """
Analyze the user's query and classify the retrieval strategy.
1. 'GRAPH': Asks about specific entities/relationships (e.g., "Who founded X?")
2. 'VECTOR': Asks for summaries/general info (e.g., "What is the history of Z?")

Extract entity names. Return JSON: {"strategy": "...", "entities": [...]}
"""


USER_TEMPLATE = """
{text}
"""


def build_messages(query: str) -> list[dict]:
    return [
        {"role": "system", "content": ROUTER_PROMPT},
        {"role": "user", "content": USER_TEMPLATE.format(text=query)},
    ]
