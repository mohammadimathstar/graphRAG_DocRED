JUDGE_PROMPT = """
You are an expert evaluator for a RAG system.
Evaluate if the 'Generated Answer' correctly answers the 'Question' based on 
the 'Ground Truth Answer' and the 'Gold Chunk'.
"""

USER_TEMPLATE = """
Question: {question}
Gold Chunk (Evidence): {gold_chunk}
Ground Truth Answer: {ground_truth_answer}
Generated Answer: {generated_answer}

Is the Generated Answer correct? (True/False) and explain why.
"""

def build_messages(query: str, 
                   gold_chunk: str, 
                   ground_truth_answer: str, 
                   generated_answer: str) -> list[dict]:
    return [
        {"role": "system", "content": JUDGE_PROMPT},
        {"role": "user", "content": USER_TEMPLATE.format(question=query, 
                                                         gold_chunk=gold_chunk, 
                                                         ground_truth_answer=ground_truth_answer, 
                                                         generated_answer=generated_answer)},
    ]
