import time
from itertools import product
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from client import OpenRouterLLM
from utils import (
    load_csv_text,
    load_dataset,
    load_metadata_schema,
    load_project_config,
    resolve_experiment_variants,
    save_json,
    write_csv,
)


def _normalize_max_token_schedule(raw_value: Any) -> List[int]:
    if isinstance(raw_value, list):
        schedule = [int(value) for value in raw_value]
    else:
        schedule = [int(raw_value)]
    cleaned = sorted({value for value in schedule if value > 0})
    if not cleaned:
        raise ValueError("config.yaml must define at least one positive max_tokens value.")
    return cleaned


def _normalize_float_list(raw_value: Any, default: float) -> List[float]:
    if raw_value is None:
        return [float(default)]
    if isinstance(raw_value, list):
        values = [float(value) for value in raw_value]
    else:
        values = [float(raw_value)]
    cleaned = list(dict.fromkeys(values))
    if not cleaned:
        raise ValueError("Evaluation hyperparameter list cannot be empty.")
    return cleaned


def _should_retry_with_more_tokens(error: Exception) -> bool:
    message = str(error).lower()
    retry_markers = [
        "length limit was reached",
        "eof while parsing",
        "finish_reason",
        "max tokens",
        "maximum context length",
        "too many tokens",
    ]
    return any(marker in message for marker in retry_markers)


def _statement_confidence(entry: Dict[str, Any]) -> float:
    confidence = float(entry.get("confidence", 0.0))
    return max(0.0, min(1.0, confidence))


def _edge_label(entry: Dict[str, Any], threshold: float) -> str:
    return "edge" if _statement_confidence(entry) > threshold else "uncertain"


def _no_edge_label(entry: Dict[str, Any], threshold: float) -> str:
    return "no_edge" if _statement_confidence(entry) > threshold else "uncertain"


def _final_decision(edge_label: str, no_edge_label: str) -> str:
    decision_map = {
        ("edge", "uncertain"): "edge",
        ("edge", "no_edge"): "uncertain",
        ("uncertain", "uncertain"): "uncertain",
        ("uncertain", "no_edge"): "no_edge",
    }
    return decision_map[(edge_label, no_edge_label)]


def _adjusted_counts(rows: List[Dict[str, Any]]) -> Dict[str, int]:
    counts = {"tp": 0, "fp": 0, "fn": 0, "tn": 0, "ue": 0, "un": 0}
    for row in rows:
        gt = row["ground_truth"]
        pred = row["final_prediction"]
        if gt == "edge" and pred == "edge":
            counts["tp"] += 1
        elif gt == "edge" and pred == "no_edge":
            counts["fn"] += 1
        elif gt == "edge" and pred == "uncertain":
            counts["ue"] += 1
        elif gt == "no_edge" and pred == "edge":
            counts["fp"] += 1
        elif gt == "no_edge" and pred == "no_edge":
            counts["tn"] += 1
        else:
            counts["un"] += 1
    return counts


def _safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def _adjusted_metrics(counts: Dict[str, int], alpha: float, beta: float) -> Dict[str, float]:
    tp = counts["tp"]
    fp = counts["fp"]
    fn = counts["fn"]
    tn = counts["tn"]
    ue = counts["ue"]
    un = counts["un"]

    precision_adj = _safe_div(tp, tp + fp + alpha * un)
    recall_adj = _safe_div(tp, tp + fn + beta * ue)
    f1_adj = _safe_div(2 * precision_adj * recall_adj, precision_adj + recall_adj)
    accuracy_adj = _safe_div(tp + tn, tp + tn + fp + fn + alpha * un + beta * ue)

    return {
        **counts,
        "precision_adj": precision_adj,
        "recall_adj": recall_adj,
        "f1_adj": f1_adj,
        "accuracy_adj": accuracy_adj,
    }


def _flatten_raw_payload(raw_payload: Dict[str, Any], gt_edges: set, threshold: float) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []

    for variant in raw_payload["variants"]:
        for model_name, model_outputs in variant["model_outputs"].items():
            edge_results = model_outputs["edge"]
            no_edge_results = model_outputs["no_edge"]
            all_edge_keys = sorted(edge_key for edge_key in set(edge_results) | set(no_edge_results) if "->" in edge_key)

            for edge_key in all_edge_keys:
                source, target = edge_key.split("->", 1)
                edge_entry = edge_results.get(edge_key, {})
                no_edge_entry = no_edge_results.get(edge_key, {})
                edge_vote = _edge_label(edge_entry, threshold)
                no_edge_vote = _no_edge_label(no_edge_entry, threshold)
                rows.append(
                    {
                        "dataset_name": raw_payload["dataset_name"],
                        "dataset_dir": raw_payload["dataset_dir"],
                        "run_id": raw_payload["run_id"],
                        "model_name": model_name,
                        "prompt_family": raw_payload["prompt_family"],
                        "prompt_style": raw_payload["prompt_style"],
                        "variant_type": variant["variant_type"],
                        "variant_name": variant["variant_name"],
                        "metadata_file": variant["metadata_file"],
                        "sampled_data_file": variant["sampled_data_file"] or "",
                        "edge_key": edge_key,
                        "source": source,
                        "target": target,
                        "ground_truth": "edge" if (source, target) in gt_edges else "no_edge",
                        "p_edge": _statement_confidence(edge_entry),
                        "p_no_edge": _statement_confidence(no_edge_entry),
                        "edge_vote": edge_vote,
                        "no_edge_vote": no_edge_vote,
                        "final_prediction": _final_decision(edge_vote, no_edge_vote),
                    }
                )

    return pd.DataFrame(rows)


def _infer_all(
    *,
    llm: OpenRouterLLM,
    dataset_dir: str,
    dataset_name: str,
    models: List[str],
    prompt_family: str,
    prompt_style: str,
    max_tokens_schedule: List[int],
    temperature: float,
    run_id: str,
) -> Dict[str, Any]:
    variants = resolve_experiment_variants(dataset_dir, prompt_family)
    raw_payload: Dict[str, Any] = {
        "run_id": run_id,
        "dataset_name": dataset_name,
        "dataset_dir": dataset_dir,
        "prompt_family": prompt_family,
        "prompt_style": prompt_style,
        "temperature": temperature,
        "max_tokens_schedule": max_tokens_schedule,
        "models": models,
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "variants": [],
    }

    for variant in variants:
        schema = load_metadata_schema(variant["metadata_file"])
        input_csv = load_csv_text(variant["sampled_data_file"]) if variant["sampled_data_file"] else ""
        variables = list(schema.get("variables", {}).keys())

        variant_payload: Dict[str, Any] = {
            "variant_type": variant["variant_type"],
            "variant_name": variant["variant_name"],
            "metadata_file": variant["metadata_file"],
            "sampled_data_file": variant["sampled_data_file"],
            "variables": variables,
            "model_outputs": {},
        }

        print(f"\nLoaded dataset: {dataset_name}")
        print(f"Variant: {variant['variant_type']} | {variant['variant_name']}")
        print(f"Variables: {variables}")

        for model in models:
            model_payload: Dict[str, Any] = {}
            for query_mode in ["edge", "no_edge"]:
                last_error: Exception | None = None
                for attempt_index, attempt_max_tokens in enumerate(max_tokens_schedule):
                    try:
                        if attempt_index == 0:
                            print(f"Running model: {model} | mode: {query_mode} | max_tokens: {attempt_max_tokens}")
                        else:
                            print(f"Retrying model: {model} | mode: {query_mode} | max_tokens: {attempt_max_tokens}")
                        model_payload[query_mode] = llm.label_all_pairs(
                            model=model,
                            schema=schema,
                            variables=variables,
                            query_mode=query_mode,
                            prompt_family=prompt_family,
                            prompt_style=prompt_style,
                            input_csv=input_csv,
                            max_tokens=attempt_max_tokens,
                            temperature=temperature,
                        )
                        last_error = None
                        break
                    except Exception as exc:
                        last_error = exc
                        has_more_attempts = attempt_index < len(max_tokens_schedule) - 1
                        if has_more_attempts and _should_retry_with_more_tokens(exc):
                            continue
                        break

                if last_error is not None:
                    raise RuntimeError(
                        f"Inference failed for model={model}, mode={query_mode}, variant={variant['variant_name']}: {last_error}"
                    )

            variant_payload["model_outputs"][model] = model_payload

        raw_payload["variants"].append(variant_payload)

    return raw_payload


def _evaluate_all(
    *,
    raw_payload: Dict[str, Any],
    gt_edges: set,
    threshold: float,
    alpha_values: List[float],
    beta_values: List[float],
) -> List[Dict[str, Any]]:
    per_edge_df = _flatten_raw_payload(raw_payload, gt_edges, threshold)
    if per_edge_df.empty:
        return []

    key_cols = [
        "dataset_name",
        "dataset_dir",
        "run_id",
        "model_name",
        "prompt_family",
        "prompt_style",
        "variant_type",
        "variant_name",
        "metadata_file",
        "sampled_data_file",
    ]

    counts_df = (
        per_edge_df.assign(
            tp=((per_edge_df["ground_truth"] == "edge") & (per_edge_df["final_prediction"] == "edge")).astype(int),
            fp=((per_edge_df["ground_truth"] == "no_edge") & (per_edge_df["final_prediction"] == "edge")).astype(int),
            fn=((per_edge_df["ground_truth"] == "edge") & (per_edge_df["final_prediction"] == "no_edge")).astype(int),
            tn=((per_edge_df["ground_truth"] == "no_edge") & (per_edge_df["final_prediction"] == "no_edge")).astype(int),
            ue=((per_edge_df["ground_truth"] == "edge") & (per_edge_df["final_prediction"] == "uncertain")).astype(int),
            un=((per_edge_df["ground_truth"] == "no_edge") & (per_edge_df["final_prediction"] == "uncertain")).astype(int),
        )
        .groupby(key_cols, as_index=False)[["tp", "fp", "fn", "tn", "ue", "un"]]
        .sum()
    )

    frames: List[pd.DataFrame] = []
    for alpha, beta in product(alpha_values, beta_values):
        frame = counts_df.copy()
        frame["threshold"] = threshold
        frame["alpha"] = alpha
        frame["beta"] = beta

        precision_denom = frame["tp"] + frame["fp"] + alpha * frame["un"]
        recall_denom = frame["tp"] + frame["fn"] + beta * frame["ue"]
        accuracy_denom = frame["tp"] + frame["tn"] + frame["fp"] + frame["fn"] + alpha * frame["un"] + beta * frame["ue"]

        frame["precision_adj"] = (frame["tp"] / precision_denom).fillna(0.0)
        frame["recall_adj"] = (frame["tp"] / recall_denom).fillna(0.0)
        f1_denom = frame["precision_adj"] + frame["recall_adj"]
        frame["f1_adj"] = ((2 * frame["precision_adj"] * frame["recall_adj"]) / f1_denom).fillna(0.0)
        frame["accuracy_adj"] = ((frame["tp"] + frame["tn"]) / accuracy_denom).fillna(0.0)
        frames.append(frame)

    result_df = pd.concat(frames, ignore_index=True)
    ordered_cols = key_cols + [
        "threshold",
        "alpha",
        "beta",
        "tp",
        "fp",
        "fn",
        "tn",
        "ue",
        "un",
        "accuracy_adj",
        "precision_adj",
        "recall_adj",
        "f1_adj",
    ]
    return result_df[ordered_cols].to_dict(orient="records")


def main() -> None:
    config = load_project_config()
    experiment = config.get("experiment", {})
    evaluation = config.get("evaluation", {})

    models = list(config.get("models", []))
    if "dataset" in config:
        dataset_dirs = [config["dataset"]]
    else:
        dataset_dirs = list(config.get("datasets", []))
    out_root = Path(str(config.get("out_root", "outputs")))
    max_tokens_schedule = _normalize_max_token_schedule(config.get("max_tokens", [8000]))
    temperature = float(config.get("temperature", 0.0))
    prompt_family = "metaData"
    prompt_style = "vanilla"
    threshold = float(evaluation.get("threshold", config.get("threshold", 0.7)))
    alpha_values = _normalize_float_list(evaluation.get("alpha", config.get("alpha")), 1.0)
    beta_values = _normalize_float_list(evaluation.get("beta", config.get("beta")), 1.0)

    if not models:
        raise ValueError("config.yaml must define a non-empty 'models' list.")
    if len(dataset_dirs) != 1:
        raise ValueError("This reproducibility package expects exactly one dataset in config.yaml.")

    dataset_dir = str(dataset_dirs[0])
    dataset_name, _, gt_edges = load_dataset(dataset_dir)
    run_id = str(config.get("run_tag") or f"run_{int(time.time())}")
    llm = OpenRouterLLM()

    raw_payload = _infer_all(
        llm=llm,
        dataset_dir=dataset_dir,
        dataset_name=dataset_name,
        models=models,
        prompt_family=prompt_family,
        prompt_style=prompt_style,
        max_tokens_schedule=max_tokens_schedule,
        temperature=temperature,
        run_id=run_id,
    )
    out_root.mkdir(parents=True, exist_ok=True)
    raw_path = out_root / "raw_llm_results.json"
    save_json(str(raw_path), raw_payload)

    evaluation_rows = _evaluate_all(
        raw_payload=raw_payload,
        gt_edges=gt_edges,
        threshold=threshold,
        alpha_values=alpha_values,
        beta_values=beta_values,
    )
    evaluation_path = out_root / "evaluation_results.csv"
    write_csv(evaluation_rows, str(evaluation_path))

    print("\n" + "=" * 60)
    print("Run finished.")
    print(f"Raw results     : {raw_path}")
    print(f"Evaluation file : {evaluation_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
