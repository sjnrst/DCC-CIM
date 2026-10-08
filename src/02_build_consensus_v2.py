from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path.cwd().resolve()
INPUT_FEATURE_DIR = ROOT / "artifacts" / "rq2_feature_tables"
OUTPUT_DIR = ROOT / "artifacts" / "rq2_feature_tables_consensus_v2"

SEED_FEATURES_INPUT = INPUT_FEATURE_DIR / "seed_alignment_features.csv"
FEATURE_COLUMNS_INPUT = INPUT_FEATURE_DIR / "feature_columns.json"

UNIVERSE_CSV = OUTPUT_DIR / "unlabeled_candidate_universe.csv"
SEED_FEATURES_V2_CSV = OUTPUT_DIR / "seed_alignment_features_consensus_v2.csv"
FEATURE_COLUMNS_V2_JSON = OUTPUT_DIR / "feature_columns_consensus_v2.json"
AUDIT_MD = OUTPUT_DIR / "feature_extraction_audit_consensus_v2.md"
MANIFEST_JSON = OUTPUT_DIR / "run_manifest_consensus_v2.json"

RUN_TS = datetime.now(timezone.utc).isoformat()
CONSENSUS_FEATURES = [
    "feat_same_doc_exact_name_count",
    "feat_same_doc_similar_name_count",
    "feat_cross_model_exact_name_count",
    "feat_cross_model_same_parent_count",
    "feat_same_doc_same_parent_count",
]
OPEN_WORLD_SOURCE_FEATURES = [
    "feat_is_phase2_candidate_source",
    "feat_is_manual_e6_source",
    "feat_is_other_source",
    "feat_is_positive_review_uncertain_source",
]
FORBIDDEN_UNIVERSE_FIELDS = {
    "binary_label",
    "error_code",
    "manual_review_final_label",
    "matched_gt_name",
    "matched_gt_id",
    "score_final",
    "score_alignment",
    "lambda_value",
}
MODEL_SOURCES = [
    ("Multi-Agent", ROOT / "private_inputs" / "model_outputs" / "multi_agent"),
    ("DeepSeek-V4-Pro", ROOT / "private_inputs" / "model_outputs" / "deepseek_v4_pro"),
    ("GLM-4.7", ROOT / "private_inputs" / "model_outputs" / "glm_4_7"),
    ("GPT-5.4", ROOT / "private_inputs" / "model_outputs" / "gpt_5_4"),
    ("GPT-5.5", ROOT / "private_inputs" / "model_outputs" / "gpt_5_5"),
    ("Qwen3.6-Plus", ROOT / "private_inputs" / "model_outputs" / "qwen_3_6_plus"),
]


def ensure_output_dir() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def extract_evidence_text(value: Any) -> str:
    if isinstance(value, dict):
        for key in ("text", "evidence_text", "content"):
            if key in value and value[key] is not None:
                return normalize_text(value[key])
        return normalize_text(json.dumps(value, ensure_ascii=False))
    if isinstance(value, list):
        parts = [extract_evidence_text(item) for item in value]
        return " ".join(part for part in parts if part)
    return normalize_text(value)


def cleaned_char_set(text: str) -> set[str]:
    return {
        char
        for char in normalize_text(text)
        if not char.isspace() and char not in "，。！？；;,.、“”\"'：:（）()[]【】/|_-"
    }


def jaccard_char(text_a: str, text_b: str) -> float:
    set_a = cleaned_char_set(text_a)
    set_b = cleaned_char_set(text_b)
    union = set_a | set_b
    if not union:
        return 0.0
    return len(set_a & set_b) / len(union)


def load_universe_rows() -> tuple[list[dict[str, str]], dict[str, int], list[str]]:
    rows: list[dict[str, str]] = []
    missing_counts = Counter()
    mapping_notes: list[str] = []
    for model_name, model_dir in MODEL_SOURCES:
        for json_path in sorted(model_dir.glob("*.json")):
            data = json.loads(json_path.read_text(encoding="utf-8"))
            doc_id = normalize_text(data.get("doc_id", json_path.stem))
            indicators = data.get("indicators", [])
            if not isinstance(indicators, list):
                mapping_notes.append(f"{rel(json_path)} indicators not list; skipped")
                continue
            for indicator in indicators:
                if not isinstance(indicator, dict):
                    continue
                pred_id = normalize_text(indicator.get("id") or indicator.get("pred_id"))
                pred_name = normalize_text(indicator.get("name") or indicator.get("pred_name"))
                pred_parent_node = normalize_text(indicator.get("parent_node") or indicator.get("pred_parent_node"))
                pred_definition = normalize_text(indicator.get("definition") or indicator.get("pred_definition"))
                pred_evidence = extract_evidence_text(indicator.get("evidence") or indicator.get("pred_evidence"))
                row = {
                    "candidate_key": f"{model_name}|{doc_id}|{pred_id}" if pred_id else f"{model_name}|{doc_id}|row-{len(rows)+1}",
                    "model_name": model_name,
                    "doc_id": doc_id,
                    "pred_id": pred_id,
                    "pred_name": pred_name,
                    "pred_parent_node": pred_parent_node,
                    "pred_definition": pred_definition,
                    "pred_evidence": pred_evidence,
                }
                for field in ("model_name", "doc_id", "pred_id", "pred_name", "pred_parent_node", "pred_definition", "pred_evidence"):
                    if not row[field]:
                        missing_counts[field] += 1
                rows.append(row)
    mapping_notes.append("JSON mapping: indicator.id -> pred_id, indicator.name -> pred_name, indicator.parent_node -> pred_parent_node, indicator.definition -> pred_definition, indicator.evidence.text -> pred_evidence.")
    mapping_notes.append("If evidence is not a dict, raw string/list content is normalized into pred_evidence.")
    return rows, dict(missing_counts), mapping_notes


def build_universe_consensus_maps(rows: list[dict[str, str]]) -> dict[str, Any]:
    same_doc_exact_name = Counter((row["doc_id"], row["pred_name"]) for row in rows)
    same_doc_same_parent = Counter((row["doc_id"], row["pred_parent_node"]) for row in rows)
    doc_name_model_sets: dict[tuple[str, str], set[str]] = defaultdict(set)
    doc_parent_model_sets: dict[tuple[str, str], set[str]] = defaultdict(set)
    rows_by_doc: dict[str, list[dict[str, str]]] = defaultdict(list)
    similar_name_count: dict[str, int] = {}

    for row in rows:
        doc_name_model_sets[(row["doc_id"], row["pred_name"])].add(row["model_name"])
        doc_parent_model_sets[(row["doc_id"], row["pred_parent_node"])].add(row["model_name"])
        rows_by_doc[row["doc_id"]].append(row)

    for doc_rows in rows_by_doc.values():
        char_sets = {row["candidate_key"]: cleaned_char_set(row["pred_name"]) for row in doc_rows}
        for row in doc_rows:
            base_key = row["candidate_key"]
            base_set = char_sets[base_key]
            count = 0
            for other in doc_rows:
                other_set = char_sets[other["candidate_key"]]
                union = base_set | other_set
                score = 0.0 if not union else len(base_set & other_set) / len(union)
                if score >= 0.6:
                    count += 1
            similar_name_count[base_key] = count

    return {
        "same_doc_exact_name": same_doc_exact_name,
        "same_doc_same_parent": same_doc_same_parent,
        "cross_model_exact_name": {key: len(value) for key, value in doc_name_model_sets.items()},
        "cross_model_same_parent": {key: len(value) for key, value in doc_parent_model_sets.items()},
        "similar_name_count": similar_name_count,
    }


def main() -> int:
    ensure_output_dir()

    universe_rows, missing_counts, mapping_notes = load_universe_rows()
    universe_fieldnames = [
        "candidate_key",
        "model_name",
        "doc_id",
        "pred_id",
        "pred_name",
        "pred_parent_node",
        "pred_definition",
        "pred_evidence",
    ]
    write_csv(UNIVERSE_CSV, universe_rows, universe_fieldnames)

    consensus = build_universe_consensus_maps(universe_rows)
    seed_rows = read_csv(SEED_FEATURES_INPUT)
    feature_columns = json.loads(FEATURE_COLUMNS_INPUT.read_text(encoding="utf-8"))
    feature_columns_v2 = [feature for feature in feature_columns if feature not in OPEN_WORLD_SOURCE_FEATURES]

    updated_seed_rows: list[dict[str, Any]] = []
    for row in seed_rows:
        updated = dict(row)
        key = f"{row['model_name']}|{row['doc_id']}|{row['pred_id']}"
        updated["feat_same_doc_exact_name_count"] = float(consensus["same_doc_exact_name"][(row["doc_id"], row["pred_name"])])
        updated["feat_same_doc_similar_name_count"] = float(consensus["similar_name_count"].get(key, 1))
        updated["feat_cross_model_exact_name_count"] = float(consensus["cross_model_exact_name"].get((row["doc_id"], row["pred_name"]), 0))
        updated["feat_cross_model_same_parent_count"] = float(consensus["cross_model_same_parent"].get((row["doc_id"], row["pred_parent_node"]), 0))
        updated["feat_same_doc_same_parent_count"] = float(consensus["same_doc_same_parent"][(row["doc_id"], row["pred_parent_node"])])
        updated_seed_rows.append(updated)

    seed_fieldnames = list(seed_rows[0].keys()) if seed_rows else []
    write_csv(SEED_FEATURES_V2_CSV, updated_seed_rows, seed_fieldnames)
    FEATURE_COLUMNS_V2_JSON.write_text(json.dumps(feature_columns_v2, ensure_ascii=False, indent=2), encoding="utf-8")

    model_counts = Counter(row["model_name"] for row in universe_rows)
    doc_count = len({row["doc_id"] for row in universe_rows})
    audit_lines = [
        "# RQ2 Consensus-v2 Feature Extraction Audit",
        "",
        f"- Generated at: {RUN_TS}",
        f"- Universe candidate count: {len(universe_rows)}",
        f"- Model coverage count: {len(model_counts)}",
        f"- Document coverage count: {doc_count}",
        f"- Seed feature rows preserved: {len(updated_seed_rows)}",
        f"- Training feature count (Consensus-v2): {len(feature_columns_v2)}",
        "",
        "## Universe Coverage",
    ]
    for model_name, count in sorted(model_counts.items()):
        audit_lines.append(f"- {model_name}: {count}")
    audit_lines.extend(["", "## Missing Field Counts"])
    for field in ["model_name", "doc_id", "pred_id", "pred_name", "pred_parent_node", "pred_definition", "pred_evidence"]:
        audit_lines.append(f"- {field}: {missing_counts.get(field, 0)}")
    audit_lines.extend(["", "## Mapping Rules"])
    for note in mapping_notes:
        audit_lines.append(f"- {note}")
    audit_lines.extend(
        [
            "",
            "## Consensus-v2 Rules",
            "- Aggregation base is `unlabeled_candidate_universe.csv`, not the frozen 347-row Seed training pool.",
            "- `feat_same_doc_exact_name_count`: count of universe candidates within the same `doc_id` whose `pred_name` exactly matches.",
            "- `feat_same_doc_similar_name_count`: count of universe candidates within the same `doc_id` whose name Jaccard-char similarity is >= 0.6.",
            "- `feat_cross_model_exact_name_count`: number of distinct models within the same `doc_id` sharing the exact same `pred_name`.",
            "- `feat_cross_model_same_parent_count`: number of distinct models within the same `doc_id` sharing the exact same `pred_parent_node`.",
            "- `feat_same_doc_same_parent_count`: count of universe candidates within the same `doc_id` sharing the exact same `pred_parent_node`.",
            "- No `binary_label`, `error_code`, `manual_review_final_label`, GT fields, or score fields are used in universe construction or in Consensus-v2 aggregation.",
        ]
    )
    AUDIT_MD.write_text("\n".join(audit_lines) + "\n", encoding="utf-8")

    manifest = {
        "created_at_utc": RUN_TS,
        "build_script": rel(Path(__file__)),
        "input_files": {
            rel(SEED_FEATURES_INPUT): sha256_file(SEED_FEATURES_INPUT),
            rel(FEATURE_COLUMNS_INPUT): sha256_file(FEATURE_COLUMNS_INPUT),
        },
        "model_source_dirs": [rel(path) for _, path in MODEL_SOURCES],
        "output_files": {},
        "counts": {
            "universe_candidate_count": len(universe_rows),
            "model_coverage_count": len(model_counts),
            "document_coverage_count": doc_count,
            "seed_feature_row_count": len(updated_seed_rows),
            "training_feature_count": len(feature_columns_v2),
        },
        "missing_counts": missing_counts,
    }
    for output_path in [UNIVERSE_CSV, SEED_FEATURES_V2_CSV, FEATURE_COLUMNS_V2_JSON, AUDIT_MD]:
        manifest["output_files"][rel(output_path)] = sha256_file(output_path)
    MANIFEST_JSON.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"universe_candidate_count={len(universe_rows)}")
    print(f"model_coverage_count={len(model_counts)}")
    print(f"document_coverage_count={doc_count}")
    print(f"training_feature_count={len(feature_columns_v2)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
