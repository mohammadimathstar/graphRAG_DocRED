SYSTEM_PROMPT = """You are an expert evaluator generating a synthetic Q&A dataset for a GraphRAG system.
You will be given a document and a list of ground truth factual triples extracted from it.
For EACH triple, generate a natural language question whose answer is the 'Object' of the triple.
You must also extract the exact, verbatim sentence from the document text that supports this triple (the gold chunk)."""

USER_PROMPT = """# Document Text
{text}

"""