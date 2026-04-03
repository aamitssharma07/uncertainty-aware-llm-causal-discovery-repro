from typing import Any, Dict

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate


SYSTEM = """You are a causal-graph labeling assistant.
Use only the provided prompt instructions, metadata, and optional sampled data.
Return a valid structured response that matches the requested schema."""

EDGE_PROMPT = """

### ROLE
    - You are an expert Causal Inference Engine with deep knowledge of Directed Acyclic Graphs (DAGs), structural equation modeling, and domain-specific mechanisms.
    - You are given the full set of variables and their definitions.
    - Analyze the provided domain and metadata to estimate the probability of a DIRECT causal relationship for every ordered pair A -> B.

### DEFINITION: "DIRECT CAUSE"
A directly causes B (A -> B) if and only if:
    1. MANIPULATION: If an external agent intervenes to change the value of A, the distribution of B changes.
    2. ADJACENCY: This effect is not entirely mediated by any other variable provided in the list. If A affects B only through C, then A -> B is false.
    3. ASYMMETRY: Changes in B do not result in changes in A through the same mechanism.

### EVALUATION CRITERIA
    - Avoid common-cause bias.
    - Avoid reverse causality.
    - Avoid proxy variables.
    - Avoid purely displaced effects.
    - Require a plausible mechanism.
    - Enforce temporal priority.

### INPUT
### METADATA
{input_json}

### TASK
- Evaluate every possible ordered pair of distinct variables in the metadata.
- For each ordered pair, provide only:
    - the probability that a direct edge A -> B exists
- Use the full variable set to reason about confounding, mediation, reverse causality, and proxy variables.
"""


NO_EDGE_PROMPT = """

### ROLE
- You are an expert Causal Inference Engine specializing in Causal Discovery and Structural Independence testing.
- Your goal is to estimate the probability that there is NO DIRECT EDGE for every ordered pair A -> B.

### TASK
Analyze the provided domain and metadata to estimate the probability of the statement: "A does NOT directly cause B."

### DEFINITION: "NO DIRECT CAUSE"
The relationship A -> B is false if any of the following are true:
1. INDEPENDENCE: Intervening on A has zero effect on the distribution of B.
2. SPURIOUSNESS: The correlation between A and B is entirely explained by a common cause C.
3. TOTAL MEDIATION: A affects B only through an intermediate variable C.
4. REVERSE ONLY: B causes A, but A does not cause B.

### EVALUATION CRITERIA for "NO EDGE"
- Identify confounders.
- Identify mediators.
- Check for proxies.
- Penalize missing mechanism.

### INPUT
### METADATA
{input_json}

### TASK
- Evaluate every possible ordered pair of distinct variables in the metadata.
- For each ordered pair, provide only:
  - the probability that the statement "A does NOT directly cause B" is true
- Use the full variable set to reason about independence, confounding, mediation, reverse-only direction, and proxy variables.
"""


def build_prompt(
    *,
    query_mode: str,
    schema: Dict[str, Any],
    prompt_family: str,
    prompt_style: str,
    attribute_a: str,
    attribute_b: str,
    input_csv: str = "",
) -> ChatPromptTemplate:
    if prompt_family != "metaData" or prompt_style != "vanilla":
        raise ValueError(
            "This reproducibility package only supports the metadata-only vanilla base prompt."
        )

    template = _pick_template(query_mode)
    rendered = _render_template(
        template,
        {
            "attribute_A": attribute_a,
            "attribute_B": attribute_b,
            "input_json": _schema_to_input_json(schema),
            "input_csv": input_csv,
        },
    )
    return ChatPromptTemplate.from_messages(
        [
            SystemMessage(content=SYSTEM),
            HumanMessage(content=rendered),
        ]
    )
def _pick_template(query_mode: str) -> str:
    if query_mode == "edge":
        return EDGE_PROMPT
    return NO_EDGE_PROMPT


def _render_template(template: str, values: Dict[str, str]) -> str:
    rendered = template
    for key, value in values.items():
        rendered = rendered.replace(f"{{{key}}}", value)
    return rendered


def _schema_to_input_json(schema: Dict[str, Any]) -> str:
    import json

    variables = [
        {"name": str(name), "description": str(desc)}
        for name, desc in (schema.get("variables", {}) or {}).items()
    ]
    obj = {
        "field": schema.get("dataset_field", ""),
        "context": schema.get("dataset_context", ""),
        "variables": variables,
    }
    return json.dumps(obj, indent=2)
