import json
from itertools import permutations
from typing import Any, Dict, List

from pydantic import BaseModel, Field, create_model


class BatchPairQueryResponse(BaseModel):
    probability: float = Field(description="Probability that the queried statement is true, between 0 and 1")


def build_batch_response_model(variables: List[str]):
    """
    Build a dynamic Pydantic model whose aliased fields are ordered-pair keys like A->B.
    """
    fields = {}
    for index, (source, target) in enumerate(permutations(variables, 2)):
        edge_key = f"{source}->{target}"
        field_name = f"edge_{index}"
        fields[field_name] = (
            BatchPairQueryResponse,
            Field(description=f"Structured result for ordered pair {edge_key}", alias=edge_key),
        )

    return create_model("BatchQueryResponse", **fields)


def _clamp_probability(value: Any) -> float:
    try:
        probability = float(value)
    except (TypeError, ValueError):
        probability = 0.5
    return max(0.0, min(1.0, probability))


def _extract_json_payload(response_text: str) -> Dict[str, Any]:
    text = response_text.strip()
    if not text:
        raise ValueError("Empty model response")

    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
        raise ValueError("Top-level JSON response is not an object")
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON object found in model response")

    parsed = json.loads(text[start : end + 1])
    if not isinstance(parsed, dict):
        raise ValueError("Extracted JSON response is not an object")
    return parsed


def _normalize_entries_mapping(entries: Dict[str, Any], variables: List[str]) -> Dict[str, Any]:
    normalized: Dict[str, Any] = {}
    for source, target in permutations(variables, 2):
        edge_key = f"{source}->{target}"
        entry = entries.get(edge_key, {}) or {}
        normalized[edge_key] = {
            "confidence": _clamp_probability(entry.get("probability", 0.5)),
            "notes": [],
        }
    return normalized


def _normalize_edges_list(edges: List[Any], variables: List[str]) -> Dict[str, Any]:
    valid_edges = {f"{source}->{target}" for source, target in permutations(variables, 2)}
    entries: Dict[str, Dict[str, Any]] = {}
    for edge in edges:
        if not isinstance(edge, dict):
            continue
        source = edge.get("source") or edge.get("from") or edge.get("attribute_a") or edge.get("a")
        target = edge.get("target") or edge.get("to") or edge.get("attribute_b") or edge.get("b")
        if not source or not target:
            continue
        edge_key = f"{source}->{target}"
        if edge_key not in valid_edges:
            continue
        probability = (
            edge.get("probability")
            if "probability" in edge
            else edge.get("confidence")
            if "confidence" in edge
            else edge.get("score")
        )
        entries[edge_key] = {"probability": probability}
    return _normalize_entries_mapping(entries, variables)


def parse_batch_query_text(response_text: str, variables: List[str]) -> Dict[str, Any]:
    payload = _extract_json_payload(response_text)

    if "edges" in payload and isinstance(payload["edges"], list):
        return _normalize_edges_list(payload["edges"], variables)

    for wrapper_key in (
        "direct_causal_relationships",
        "direct_causal_relations",
        "relations",
        "predictions",
        "results",
    ):
        wrapped = payload.get(wrapper_key)
        if isinstance(wrapped, dict):
            return _normalize_entries_mapping(wrapped, variables)
        if isinstance(wrapped, list):
            return _normalize_edges_list(wrapped, variables)

    return _normalize_entries_mapping(payload, variables)


def normalize_batch_query_output(
    payload: BaseModel,
    variables: List[str],
) -> Dict[str, Any]:
    """
    Convert a structured all-pairs response into the project's raw-results format.
    """
    if hasattr(payload, "model_dump"):
        entries = payload.model_dump(by_alias=True)
    else:
        entries = payload.dict(by_alias=True)

    return _normalize_entries_mapping(entries, variables)
