from __future__ import annotations

import json
from pathlib import Path

from rest_server.features.ask_patra.models import AskPatraStarter


DEFAULT_SYSTEM_PROMPT = """You are Ask Patra, a concise PATRA assistant.
Your job is to help users discover model cards and datasheets, summarize what PATRA can do, and answer using only the provided context.
Do not invent records or metadata. If relevant records are not found, say that directly and suggest a narrower search.
Always prefer short, direct answers. Mention matching records by title when useful."""


DEFAULT_BEHAVIOR_PROMPT = """When citations are provided, ground the answer in them.
If the user asks what PATRA can help with, mention:
- browse model cards
- browse datasheets
- compare records
- inspect resource metadata
- find likely matches by keyword
- explain PATRA workflows such as Agent Toolkit, Edit Records, and Automated Ingestion
If the user asks for a lookup, summarize the most relevant records first, then point to the routes."""


DEFAULT_STARTER_PROMPTS = [
    AskPatraStarter(title="Look up model cards", prompt="Find model cards related to crop yield forecasting."),
    AskPatraStarter(title="Look up datasheets", prompt="Find datasheets related to geospatial or agricultural datasets."),
    AskPatraStarter(title="Explain PATRA", prompt="What can you help me do inside PATRA?"),
    AskPatraStarter(title="Compare resources", prompt="Show me relevant model cards and datasheets for weather-driven prediction."),
]


def ensure_prompt_templates(prompts_dir: Path) -> list[AskPatraStarter]:
    prompts_dir.mkdir(parents=True, exist_ok=True)
    system_path = prompts_dir / "system_prompt.md"
    behavior_path = prompts_dir / "behavior_prompt.md"
    starter_path = prompts_dir / "starter_prompts.json"

    if not system_path.exists():
        system_path.write_text(DEFAULT_SYSTEM_PROMPT, encoding="utf-8")
    if not behavior_path.exists():
        behavior_path.write_text(DEFAULT_BEHAVIOR_PROMPT, encoding="utf-8")
    if not starter_path.exists():
        starter_path.write_text(
            json.dumps([item.model_dump() for item in DEFAULT_STARTER_PROMPTS], indent=2),
            encoding="utf-8",
        )
    try:
        loaded = json.loads(starter_path.read_text(encoding="utf-8"))
        return [AskPatraStarter.model_validate(item) for item in loaded]
    except Exception:
        return DEFAULT_STARTER_PROMPTS

