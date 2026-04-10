from __future__ import annotations

from rest_server.features.intent_schema.models import IntentSchemaStarter


SYSTEM_PROMPT = """You are PATRA Intent Schema, an assistant that converts modeling intent into a structured target schema.

Your job:
- interpret the user's modeling objective,
- infer a realistic tabular prediction setup,
- define a compact schema with clear boundaries,
- and surface ambiguity instead of pretending certainty.

Rules:
- Output must be valid JSON only.
- All text values in the JSON must be written in English.
- Keep the schema compact. Respect the requested max_fields.
- Prefer realistic tabular ML fields over abstract business jargon.
- Do not invent impossible certainty. If the task is underspecified, add ambiguity_warnings and assumptions.
- Always define a single target_column.
- For value ranges and distribution expectations, be practical and concise.
- Do not propose PII unless it is unavoidable, and call it out in notes when relevant."""


STARTER_PROMPTS = [
    IntentSchemaStarter(title="Customer Churn", prompt="I want to build a model for predicting customer churn."),
    IntentSchemaStarter(title="Crop Yield", prompt="I want to build a model for predicting crop yield."),
    IntentSchemaStarter(title="Equipment Failure", prompt="I want to build a model for predicting equipment failure risk."),
    IntentSchemaStarter(title="Wildlife Detection", prompt="I want to build a model for wildlife image detection and filtering."),
]


def build_generation_prompt(*, intent_text: str, context: str | None, max_fields: int) -> str:
    context_block = context.strip() if context and context.strip() else "No additional context"
    return f"""
Convert the modeling intent below into a structured target schema.

User intent:
{intent_text.strip()}

Additional context:
{context_block}

Requirements:
1. Assume this is an early tabular ML task and choose one of classification / regression / forecasting / ranking when possible.
2. Return:
   - intent_summary
   - task_type
   - entity_grain
   - target_column
   - label_definition
   - prediction_horizon
   - ambiguity_warnings
   - assumptions
   - schema_fields
3. Return at most {max_fields} schema fields.
4. Every field must include:
   - name
   - data_type
   - semantic_role
   - description
   - expected_range
   - distribution_expectation
   - required
   - notes
5. semantic_role must be one of:
   - target
   - feature
   - identifier
   - timestamp
   - grouping
   - unknown
6. If the intent is underspecified, do not fake certainty. Put unresolved issues in ambiguity_warnings and assumptions.
7. All natural-language text in the JSON must be in English.

Return JSON only. Do not return Markdown. Do not wrap the result in a code block.
""".strip()
