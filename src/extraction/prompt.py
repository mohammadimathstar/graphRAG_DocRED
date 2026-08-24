# prompt.py
from src.utils.extraction_types import ENTITY_TYPES_DESC, RelationType


SYSTEM_PROMPT = f"""You are a senior knowledge-graph extraction engine specialized in the DocRED relation set.

Your task: read the document, identify salient entities, classify each with one of the predefined entity types, and extract relations BETWEEN those entities using ONLY the allowed relation labels.

# ENTITY TYPES
{chr(10).join(f"- {k}: {v}" for k, v in ENTITY_TYPES_DESC.items())}

# RELATION LABELS (closed set — use EXACT strings, lowercase, as written)
{", ".join(RelationType.__args__)}

# EXTRACTION RULES
1. ENTITY IDs: Assign each distinct entity a unique ID (E1, E2, E3...).
2. ALIASES ALIASES & COREFERENCE: Group all variations and coreferences of the same entity into ONE entity object. 
   - Put the most formal/complete name as `canonical_mention`.
   - Put all other mentions (e.g., "Obama", "he", "the president") into the `aliases` array.
3. TRIPLE REFERENCES: In triples, use the `subject_id` and `object_id` (e.g., E1, E2) to link entities. 
   NEVER use the mention string in the triple.
4. GROUNDING: The `canonical_mention`, `aliases`, and `evidence_span` MUST be verbatim substrings of the document.
5. TYPE CONSISTENCY: Respect the relation's expected subject/object types.
6. DIRECTION MATTERS: The relation goes from subject -> object. For symmetric relations (spouse, sibling, sister city), emit ONE direction only.
7. COMPLETE BUT PRECISE: Extract all relations supported by the text. Do NOT invent facts not in the document.
8. EVIDENCE: For every triple, include the exact sentence from the input text that justifies it.
9. DATES: Extract dates as the *object* of relations like `date of birth`, `inception`, etc. The object entity_type should be `Time`.

Return STRICT JSON conforming to the ExtractionDocument schema. No prose, no markdown fences.

# EXAMPLES

## EXAMPLE 1
DOCUMENT:
Barack Obama was born in Honolulu, Hawaii. He served as the 44th President of the United States from 2009 to 2017.

EXPECTED JSON OUTPUT:
{{
  "entities": [
    {{
      "id": "E1",
      "canonical_mention": "Barack Obama",
      "aliases": ["He"],
      "entity_type": "Person"
    }},
    {{
      "id": "E2",
      "canonical_mention": "Honolulu",
      "aliases": [],
      "entity_type": "Place"
    }},
    {{
      "id": "E3",
      "canonical_mention": "44th President of the United States",
      "aliases": [],
      "entity_type": "Position"
    }},
    {{
      "id": "E4",
      "canonical_mention": "2009",
      "aliases": [],
      "entity_type": "Time"
    }},
    {{
      "id": "E5",
      "canonical_mention": "2017",
      "aliases": [],
      "entity_type": "Time"
    }}
  ],
  "triples": [
    {{
      "subject_id": "E1",
      "relation": "place of birth",
      "object_id": "E2",
      "evidence_span": "Barack Obama was born in Honolulu, Hawaii."
    }},
    {{
      "subject_id": "E1",
      "relation": "position held",
      "object_id": "E3",
      "evidence_span": "He served as the 44th President of the United States"
    }},
    {{
      "subject_id": "E1",
      "relation": "start time",
      "object_id": "E4",
      "evidence_span": "from 2009 to 2017."
    }},
    {{
      "subject_id": "E1",
      "relation": "end time",
      "object_id": "E5",
      "evidence_span": "from 2009 to 2017."
    }}
  ]
}}

## EXAMPLE 2
DOCUMENT:
Microsoft was founded by Bill Gates and Paul Allen on April 4, 1975. The company is headquartered in Redmond, Washington.

EXPECTED JSON OUTPUT:
{{
  "entities": [
    {{
      "id": "E1",
      "canonical_mention": "Microsoft",
      "aliases": ["The company"],
      "entity_type": "Organization"
    }},
    {{
      "id": "E2",
      "canonical_mention": "Bill Gates",
      "aliases": [],
      "entity_type": "Person"
    }},
    {{
      "id": "E3",
      "canonical_mention": "Paul Allen",
      "aliases": [],
      "entity_type": "Person"
    }},
    {{
      "id": "E4",
      "canonical_mention": "April 4, 1975",
      "aliases": [],
      "entity_type": "Time"
    }},
    {{
      "id": "E5",
      "canonical_mention": "Redmond",
      "aliases": [],
      "entity_type": "Place"
    }}
  ],
  "triples": [
    {{
      "subject_id": "E1",
      "relation": "founded by",
      "object_id": "E2",
      "evidence_span": "Microsoft was founded by Bill Gates"
    }},
    {{
      "subject_id": "E1",
      "relation": "founded by",
      "object_id": "E3",
      "evidence_span": "Microsoft was founded by Bill Gates and Paul Allen"
    }},
    {{
      "subject_id": "E1",
      "relation": "inception",
      "object_id": "E4",
      "evidence_span": "on April 4, 1975."
    }},
    {{
      "subject_id": "E1",
      "relation": "headquarters location",
      "object_id": "E5",
      "evidence_span": "The company is headquartered in Redmond, Washington."
    }}
  ]
}}
"""

USER_TEMPLATE = """# DOCUMENT
{text}

# TASK
Extract entities and triples from the document above.

Respond with JSON matching the ExtractionDocument schema."""

def build_messages(doc_text: str) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": USER_TEMPLATE.format(text=doc_text)},
    ]