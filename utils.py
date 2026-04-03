import csv
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


# Shared project config lives next to the code so every entrypoint reads the same file.
CONFIG_PATH = Path(__file__).with_name("config.yaml")
PROJECT_DIR = CONFIG_PATH.parent
REPO_ROOT = PROJECT_DIR.parent


def load_project_config() -> Dict[str, Any]:
    """Load project-level config.yaml from this folder."""
    if not CONFIG_PATH.exists():
        return {}
    with CONFIG_PATH.open("r") as f:
        data = yaml.safe_load(f) or {}
    return data if isinstance(data, dict) else {}


def load_dataset(dataset_dir: str):
    """Load dataset description and ground truth graph."""
    dataset_path = _resolve_dataset_dir(dataset_dir)
    name = dataset_path.name
    desc_path = dataset_path / f"{name}_description.json"
    graph_path = dataset_path / f"{name}_graph.json"

    with open(desc_path, "r") as f:
        desc = json.load(f)
    with open(graph_path, "r") as f:
        graph = json.load(f)

    # Convert the raw dataset files into the schema format expected by prompting code.
    schema = {
        "dataset_field": desc.get("field", ""),
        "dataset_context": desc.get("context", ""),
        "variables": {v["name"]: v.get("description", "") for v in desc.get("variables", [])},
    }
    gt_edges = {(e["source"], e["target"]) for e in graph.get("edges", [])}
    return name, schema, gt_edges


def load_metadata_schema(metadata_file: str | Path) -> Dict[str, Any]:
    metadata_path = Path(metadata_file)
    with metadata_path.open("r") as f:
        desc = json.load(f)

    return {
        "dataset_field": desc.get("field", ""),
        "dataset_context": desc.get("context", ""),
        "variables": {v["name"]: v.get("description", "") for v in desc.get("variables", [])},
    }


def load_csv_text(csv_file: str | Path) -> str:
    return Path(csv_file).read_text().strip()


def resolve_experiment_variants(dataset_dir: str, prompt_family: str) -> List[Dict[str, Optional[str]]]:
    dataset_path = _resolve_dataset_dir(dataset_dir)
    dataset_name = dataset_path.name
    metadata_clean_dir = dataset_path / "MetaData" / "clean"
    metadata_noisy_dir = dataset_path / "MetaData" / "noisy"
    sampled_clean_dir = dataset_path / "sampledData" / "clean"
    sampled_corrupt_dir = dataset_path / "sampledData" / "corrupt"

    metadata_clean_files = sorted(metadata_clean_dir.glob("*.json"))
    metadata_noisy_files = sorted(metadata_noisy_dir.glob("*.json"))
    sampled_clean_files = sorted(sampled_clean_dir.glob("*.csv"))
    sampled_corrupt_files = sorted(sampled_corrupt_dir.glob("*.csv"))

    if prompt_family == "metaData":
        variants = [
            _build_variant(
                dataset_name=dataset_name,
                variant_type="metadata_clean",
                metadata_file=metadata_file,
                sampled_data_file=None,
            )
            for metadata_file in metadata_clean_files
        ]
        variants.extend(
            _build_variant(
                dataset_name=dataset_name,
                variant_type="metadata_noisy",
                metadata_file=metadata_file,
                sampled_data_file=None,
            )
            for metadata_file in metadata_noisy_files
        )
        return variants

    if prompt_family == "dataMetaData":
        variants = []
        for metadata_file in metadata_clean_files:
            for sampled_data_file in [*sampled_clean_files, *sampled_corrupt_files]:
                variant_type = "sampled_clean" if sampled_data_file in sampled_clean_files else "sampled_corrupt"
                variants.append(
                    _build_variant(
                        dataset_name=dataset_name,
                        variant_type=variant_type,
                        metadata_file=metadata_file,
                        sampled_data_file=sampled_data_file,
                    )
                )
        return variants

    raise ValueError(f"Unsupported prompt_family: {prompt_family}")


def _build_variant(
    *,
    dataset_name: str,
    variant_type: str,
    metadata_file: Path,
    sampled_data_file: Optional[Path],
) -> Dict[str, Optional[str]]:
    metadata_stem = metadata_file.stem
    sampled_stem = sampled_data_file.stem if sampled_data_file else None
    variant_name = metadata_stem if sampled_stem is None else f"{metadata_stem}__{sampled_stem}"
    return {
        "dataset_name": dataset_name,
        "variant_type": variant_type,
        "variant_name": variant_name,
        "metadata_file": str(metadata_file),
        "sampled_data_file": str(sampled_data_file) if sampled_data_file else None,
    }


def _resolve_dataset_dir(dataset_dir: str) -> Path:
    """
    Resolve dataset paths relative to this folder or the repo root.
    If the configured relative path is stale, fall back to searching by dataset name.
    """
    raw_path = Path(dataset_dir)
    candidates = []

    if raw_path.is_absolute():
        candidates.append(raw_path)
    else:
        candidates.extend(
            [
                PROJECT_DIR / raw_path,
                REPO_ROOT / raw_path,
                Path.cwd() / raw_path,
            ]
        )

    for candidate in candidates:
        if candidate.is_dir():
            return candidate.resolve()

    dataset_name = raw_path.name
    for candidate in REPO_ROOT.rglob(dataset_name):
        if not candidate.is_dir():
            continue
        desc_path = candidate / f"{dataset_name}_description.json"
        graph_path = candidate / f"{dataset_name}_graph.json"
        if desc_path.is_file() and graph_path.is_file():
            return candidate.resolve()

    checked = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(
        f"Could not find dataset directory '{dataset_dir}'. Checked: {checked}"
    )


def write_csv(rows: List[Dict[str, Any]], path: str):
    """Write rows to a new CSV file."""
    cols = sorted({k for r in rows for k in r.keys()})
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)


def append_csv(rows: List[Dict[str, Any]], path: str):
    """Append rows to a CSV, merging any new columns with NaN for old rows."""
    import pandas as pd

    os.makedirs(os.path.dirname(path), exist_ok=True)
    new_df = pd.DataFrame(rows)
    if os.path.exists(path):
        existing = pd.read_csv(path)
        combined = pd.concat([existing, new_df], ignore_index=True)
    else:
        combined = new_df
    combined.to_csv(path, index=False)


def save_json(path: str, obj: Any):
    """Save object as JSON file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
