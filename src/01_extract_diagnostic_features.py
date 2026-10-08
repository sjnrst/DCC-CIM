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
INPUT_DIR = ROOT / "artifacts" / "rq2_dual_track_dataset"
OUTPUT_DIR = ROOT / "artifacts" / "rq2_feature_tables"

SEED_INPUT = INPUT_DIR / "seed_alignment_train_pool.csv"
OPEN_WORLD_INPUT = INPUT_DIR / "open_world_stability_pool.csv"
BENCHMARK_INPUT = INPUT_DIR / "benchmark_conflict_challenge_16.csv"
REVIEW_REMOVED_INPUT = INPUT_DIR / "positive_review_removed_challenge.csv"
FOLDS_INPUT = INPUT_DIR / "doc_group_folds_seed42.json"
GUARD_INPUT = INPUT_DIR / "feature_leakage_guard.json"

SEED_OUTPUT = OUTPUT_DIR / "seed_alignment_features.csv"
OPEN_WORLD_OUTPUT = OUTPUT_DIR / "open_world_stability_features.csv"
CHALLENGE_OUTPUT = OUTPUT_DIR / "challenge_features.csv"
FEATURE_COLUMNS_OUTPUT = OUTPUT_DIR / "feature_columns.json"
GUARD_V2_OUTPUT = OUTPUT_DIR / "feature_leakage_guard_v2.json"
FEATURE_SUMMARY_OUTPUT = OUTPUT_DIR / "feature_summary.csv"
AUDIT_OUTPUT = OUTPUT_DIR / "feature_extraction_audit.md"
VALIDATION_OUTPUT = OUTPUT_DIR / "validation_results.json"
MANIFEST_OUTPUT = OUTPUT_DIR / "run_manifest.json"

RUN_TS = datetime.now(timezone.utc).isoformat()

FORBIDDEN_FEATURE_FIELDS = [
    "candidate_key",
    "model_name",
    "doc_id",
    "pred_id",
    "matched_gt_id",
    "matched_gt_name",
    "all_matched_gt_ids",
    "all_matched_gt_names",
    "matched_gt_ids",
    "matched_gt_names",
    "best_gt_name",
    "best_gt_parent_node",
    "gt_parent_node",
    "score_alignment",
    "lambda_value",
    "score_final",
    "binary_label",
    "training_role",
    "supervision_label",
    "supervision_source",
    "source_pool",
    "source_match_file",
    "source_evidence_file",
    "model_output_file",
    "error_code",
    "candidate_final_label",
    "candidate_final_name",
    "manual_review_final_label",
    "annotator_1_label",
    "annotator_2_label",
    "agreement_status",
    "adjudication_reason",
    "candidate_decision",
    "label_quality_flag",
    "label_quality_note",
    "positive_pool_status",
    "review_source",
    "challenge_type",
    "analysis_branch",
    "strict_training_eligible",
]

COMMON_FEATURE_COLUMNS = [
    "feat_pred_name_char_len",
    "feat_definition_char_len",
    "feat_evidence_char_len",
    "feat_parent_node_char_len",
    "feat_definition_sentence_count",
    "feat_evidence_sentence_count",
    "feat_name_evidence_char_overlap",
    "feat_name_definition_char_overlap",
    "feat_definition_evidence_char_overlap",
    "feat_name_evidence_jaccard_char",
    "feat_name_definition_jaccard_char",
    "feat_definition_evidence_jaccard_char",
    "feat_definition_has_eval_object",
    "feat_definition_has_eval_direction",
    "feat_definition_has_degree_word",
    "feat_definition_has_compliance_word",
    "feat_definition_has_consistency_word",
    "feat_definition_has_risk_word",
    "feat_definition_has_empty_or_vague_pattern",
    "feat_definition_parallel_concept_count",
    "feat_evidence_pass_bool",
    "feat_evidence_lcs_ratio",
    "feat_evidence_too_short",
    "feat_evidence_too_long",
    "feat_evidence_contains_policy_action",
    "feat_evidence_contains_normative_word",
    "feat_parent_depth",
    "feat_parent_has_value_core",
    "feat_parent_has_cultural_adaptation",
    "feat_parent_has_scenario_support",
    "feat_parent_has_known_level1",
    "feat_parent_contains_pred_name",
    "feat_parent_path_char_len",
    "feat_same_doc_exact_name_count",
    "feat_same_doc_similar_name_count",
    "feat_cross_model_exact_name_count",
    "feat_cross_model_same_parent_count",
    "feat_same_doc_same_parent_count",
]

OPEN_WORLD_EXTRA_FEATURE_COLUMNS = [
    "feat_cluster_frequency",
    "feat_has_cluster_id",
    "feat_open_world_name_reuse_count",
    "feat_open_world_parent_reuse_count",
    "feat_open_world_doc_frequency",
    "feat_open_world_cross_model_frequency",
    "feat_is_phase2_candidate_source",
    "feat_is_manual_e6_source",
    "feat_is_other_source",
    "feat_is_positive_review_uncertain_source",
]

ALL_FEATURE_COLUMNS = COMMON_FEATURE_COLUMNS + OPEN_WORLD_EXTRA_FEATURE_COLUMNS

SEED_METADATA_COLUMNS = [
    "candidate_key",
    "model_name",
    "doc_id",
    "pred_id",
    "binary_label",
    "training_role",
    "analysis_branch",
    "source_pool",
    "error_code",
    "pred_name",
    "pred_definition",
    "pred_parent_node",
    "pred_evidence",
]

OPEN_WORLD_METADATA_COLUMNS = [
    "candidate_key",
    "model_name",
    "doc_id",
    "pred_id",
    "analysis_branch",
    "training_role",
    "open_world_status",
    "source_pool",
    "pred_name",
    "pred_definition",
    "pred_parent_node",
    "pred_evidence",
    "cluster_id",
    "cluster_frequency",
]

CHALLENGE_METADATA_COLUMNS = [
    "candidate_key",
    "model_name",
    "doc_id",
    "pred_id",
    "analysis_branch",
    "training_role",
    "challenge_type",
    "source_pool",
    "pred_name",
    "pred_definition",
    "pred_parent_node",
    "pred_evidence",
]

SUPERVISION_COLUMNS = ["binary_label", "training_role", "error_code", "open_world_status", "challenge_type", "source_pool", "analysis_branch"]
FOLD_COLUMNS = [f"fold_{idx}_role" for idx in range(5)]

DEGREE_WORDS = ("程度", "水平", "比例", "深度", "广度", "频率", "能力", "强度", "质量", "效能", "效果", "覆盖")
COMPLIANCE_WORDS = ("合规", "依法", "规范", "制度", "机制", "标准", "要求", "守法", "整改", "治理")
CONSISTENCY_WORDS = ("一致", "协同", "协调", "统一", "兼容", "衔接", "联动", "协作", "融合")
RISK_WORDS = ("风险", "安全", "隐患", "偏见", "歧视", "失衡", "滥用", "违法", "违规", "冲突")
POLICY_ACTION_WORDS = ("推进", "加强", "建立", "健全", "鼓励", "支持", "开展", "促进", "完善", "规范", "提升", "落实")
NORMATIVE_WORDS = ("应", "应当", "不得", "禁止", "要求", "坚持", "规范", "依法", "必须")
VAGUE_PATTERNS = ("相关", "等", "等等", "能力建设", "工作推进", "相关内容", "有关内容")


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


def to_float(value: Any, default: float = 0.0) -> float:
    text = normalize_text(value)
    if not text:
        return default
    try:
        return float(text)
    except ValueError:
        lowered = text.lower()
        if lowered == "true":
            return 1.0
        if lowered == "false":
            return 0.0
        return default


def to_int(value: Any, default: int = 0) -> int:
    return int(round(to_float(value, float(default))))


def count_sentences(text: str) -> int:
    stripped = normalize_text(text)
    if not stripped:
        return 0
    parts = [part for part in re.split(r"[。！？!?；;]+", stripped) if part.strip()]
    return len(parts) if parts else 1


def cleaned_char_set(text: str) -> set[str]:
    return {char for char in normalize_text(text) if not char.isspace() and char not in "，。！？；;,.、“”\"'：:（）()[]【】/|_-"}


def overlap_count(text_a: str, text_b: str) -> int:
    return len(cleaned_char_set(text_a) & cleaned_char_set(text_b))


def jaccard_char(text_a: str, text_b: str) -> float:
    set_a = cleaned_char_set(text_a)
    set_b = cleaned_char_set(text_b)
    union = set_a | set_b
    if not union:
        return 0.0
    return len(set_a & set_b) / len(union)


def contains_any(text: str, words: tuple[str, ...]) -> int:
    normalized = normalize_text(text)
    return int(any(word in normalized for word in words))


def count_parallel_concepts(text: str) -> int:
    normalized = normalize_text(text)
    if not normalized:
        return 0
    return normalized.count("、") + normalized.count("及") + normalized.count("与") + normalized.count("和")


def definition_has_eval_object(text: str) -> int:
    normalized = normalize_text(text)
    return int(("评估" in normalized or "衡量" in normalized) and ("对" in normalized or "主体" in normalized or "语料" in normalized))


def definition_has_eval_direction(text: str) -> int:
    normalized = normalize_text(text)
    return int(any(token in normalized for token in ("程度", "比例", "水平", "能力", "深度", "强度", "关注度", "完善度")))


def definition_has_vague_pattern(text: str) -> int:
    normalized = normalize_text(text)
    if not normalized:
        return 1
    if len(normalized) <= 6:
        return 1
    return int(any(pattern in normalized for pattern in VAGUE_PATTERNS))


def parent_depth(text: str) -> int:
    normalized = normalize_text(text)
    if not normalized:
        return 0
    return len([part for part in normalized.split("/") if part.strip()])


def build_fold_role_map(folds: list[dict[str, Any]]) -> dict[str, dict[str, str]]:
    role_map: dict[str, dict[str, str]] = defaultdict(dict)
    for fold in folds:
        fold_index = int(fold["fold_index"])
        for doc_id in fold["train_doc_ids"]:
            role_map[doc_id][f"fold_{fold_index}_role"] = "train"
        for doc_id in fold["test_doc_ids"]:
            role_map[doc_id][f"fold_{fold_index}_role"] = "test"
    return role_map


def add_fold_roles(row: dict[str, str], role_map: dict[str, dict[str, str]]) -> None:
    doc_roles = role_map.get(row.get("doc_id", ""), {})
    for field in FOLD_COLUMNS:
        row[field] = doc_roles.get(field, "train")


def build_consensus_maps(rows: list[dict[str, str]]) -> dict[str, Any]:
    same_doc_exact_name = Counter((row.get("doc_id", ""), row.get("pred_name", "")) for row in rows)
    same_doc_same_parent = Counter((row.get("doc_id", ""), row.get("pred_parent_node", "")) for row in rows)
    cross_model_exact_name: dict[str, int] = {}
    cross_model_same_parent: dict[str, int] = {}
    for name, group in defaultdict(list).items():
        pass
    name_models: dict[str, set[str]] = defaultdict(set)
    parent_models: dict[str, set[str]] = defaultdict(set)
    open_world_name_rows = Counter(row.get("pred_name", "") for row in rows)
    open_world_parent_rows = Counter(row.get("pred_parent_node", "") for row in rows)
    name_doc_frequency: dict[str, set[str]] = defaultdict(set)
    name_model_frequency: dict[str, set[str]] = defaultdict(set)
    similar_name_count: dict[str, int] = {}
    rows_by_doc: dict[str, list[dict[str, str]]] = defaultdict(list)

    for row in rows:
        name = row.get("pred_name", "")
        parent = row.get("pred_parent_node", "")
        model = row.get("model_name", "")
        doc_id = row.get("doc_id", "")
        candidate_key = row.get("candidate_key", "")
        name_models[name].add(model)
        parent_models[parent].add(model)
        name_doc_frequency[name].add(doc_id)
        name_model_frequency[name].add(model)
        rows_by_doc[doc_id].append(row)
        similar_name_count[candidate_key] = 0

    for name, models in name_models.items():
        cross_model_exact_name[name] = len(models)
    for parent, models in parent_models.items():
        cross_model_same_parent[parent] = len(models)

    for doc_rows in rows_by_doc.values():
        char_sets = {row["candidate_key"]: cleaned_char_set(row.get("pred_name", "")) for row in doc_rows}
        for row in doc_rows:
            key = row["candidate_key"]
            base_set = char_sets[key]
            count = 0
            for other in doc_rows:
                other_key = other["candidate_key"]
                if other_key == key:
                    continue
                other_set = char_sets[other_key]
                union = base_set | other_set
                score = 0.0 if not union else len(base_set & other_set) / len(union)
                if score >= 0.5:
                    count += 1
            similar_name_count[key] = count

    return {
        "same_doc_exact_name": same_doc_exact_name,
        "same_doc_same_parent": same_doc_same_parent,
        "cross_model_exact_name": cross_model_exact_name,
        "cross_model_same_parent": cross_model_same_parent,
        "similar_name_count": similar_name_count,
        "open_world_name_rows": open_world_name_rows,
        "open_world_parent_rows": open_world_parent_rows,
        "name_doc_frequency": {name: len(docs) for name, docs in name_doc_frequency.items()},
        "name_model_frequency": {name: len(models) for name, models in name_model_frequency.items()},
    }


def compute_common_features(row: dict[str, str], consensus: dict[str, Any]) -> dict[str, float]:
    name = row.get("pred_name", "")
    definition = row.get("pred_definition", "")
    evidence = row.get("pred_evidence", "")
    parent = row.get("pred_parent_node", "")
    candidate_key = row.get("candidate_key", "")
    doc_id = row.get("doc_id", "")

    features: dict[str, float] = {
        "feat_pred_name_char_len": float(len(normalize_text(name))),
        "feat_definition_char_len": float(len(normalize_text(definition))),
        "feat_evidence_char_len": float(len(normalize_text(evidence))),
        "feat_parent_node_char_len": float(len(normalize_text(parent))),
        "feat_definition_sentence_count": float(count_sentences(definition)),
        "feat_evidence_sentence_count": float(count_sentences(evidence)),
        "feat_name_evidence_char_overlap": float(overlap_count(name, evidence)),
        "feat_name_definition_char_overlap": float(overlap_count(name, definition)),
        "feat_definition_evidence_char_overlap": float(overlap_count(definition, evidence)),
        "feat_name_evidence_jaccard_char": float(jaccard_char(name, evidence)),
        "feat_name_definition_jaccard_char": float(jaccard_char(name, definition)),
        "feat_definition_evidence_jaccard_char": float(jaccard_char(definition, evidence)),
        "feat_definition_has_eval_object": float(definition_has_eval_object(definition)),
        "feat_definition_has_eval_direction": float(definition_has_eval_direction(definition)),
        "feat_definition_has_degree_word": float(contains_any(definition, DEGREE_WORDS)),
        "feat_definition_has_compliance_word": float(contains_any(definition, COMPLIANCE_WORDS)),
        "feat_definition_has_consistency_word": float(contains_any(definition, CONSISTENCY_WORDS)),
        "feat_definition_has_risk_word": float(contains_any(definition, RISK_WORDS)),
        "feat_definition_has_empty_or_vague_pattern": float(definition_has_vague_pattern(definition)),
        "feat_definition_parallel_concept_count": float(count_parallel_concepts(definition)),
        "feat_evidence_pass_bool": float(int(normalize_text(row.get("evidence_pass", "")).lower() == "true")),
        "feat_evidence_lcs_ratio": float(to_float(row.get("evidence_lcs_ratio", ""), 0.0)),
        "feat_evidence_too_short": float(int(len(normalize_text(evidence)) < 20)),
        "feat_evidence_too_long": float(int(len(normalize_text(evidence)) > 180)),
        "feat_evidence_contains_policy_action": float(contains_any(evidence, POLICY_ACTION_WORDS)),
        "feat_evidence_contains_normative_word": float(contains_any(evidence, NORMATIVE_WORDS)),
        "feat_parent_depth": float(parent_depth(parent)),
        "feat_parent_has_value_core": float(int("价值核心维度" in parent)),
        "feat_parent_has_cultural_adaptation": float(int("文化适配维度" in parent)),
        "feat_parent_has_scenario_support": float(int("场景支撑维度" in parent)),
        "feat_parent_has_known_level1": float(int(any(token in parent for token in ("价值核心维度", "文化适配维度", "场景支撑维度")))),
        "feat_parent_contains_pred_name": float(int(normalize_text(name) != "" and normalize_text(name) in normalize_text(parent))),
        "feat_parent_path_char_len": float(len(normalize_text(parent))),
        "feat_same_doc_exact_name_count": float(consensus["same_doc_exact_name"][(doc_id, name)]),
        "feat_same_doc_similar_name_count": float(consensus["similar_name_count"].get(candidate_key, 0)),
        "feat_cross_model_exact_name_count": float(consensus["cross_model_exact_name"].get(name, 0)),
        "feat_cross_model_same_parent_count": float(consensus["cross_model_same_parent"].get(parent, 0)),
        "feat_same_doc_same_parent_count": float(consensus["same_doc_same_parent"][(doc_id, parent)]),
    }
    return features


def compute_open_world_extra_features(row: dict[str, str], consensus: dict[str, Any]) -> dict[str, float]:
    source_pool = row.get("source_pool", "")
    error_code = row.get("error_code", "")
    open_world_status = row.get("open_world_status", "")
    name = row.get("pred_name", "")
    parent = row.get("pred_parent_node", "")
    features = {
        "feat_cluster_frequency": float(to_float(row.get("cluster_frequency", ""), 0.0)),
        "feat_has_cluster_id": float(int(normalize_text(row.get("cluster_id", "")) != "")),
        "feat_open_world_name_reuse_count": float(consensus["open_world_name_rows"].get(name, 0)),
        "feat_open_world_parent_reuse_count": float(consensus["open_world_parent_rows"].get(parent, 0)),
        "feat_open_world_doc_frequency": float(consensus["name_doc_frequency"].get(name, 0)),
        "feat_open_world_cross_model_frequency": float(consensus["name_model_frequency"].get(name, 0)),
        "feat_is_phase2_candidate_source": float(int(source_pool == "phase2_items")),
        "feat_is_manual_e6_source": float(int(error_code == "E6" or open_world_status == "stability_insufficient")),
        "feat_is_other_source": float(int(error_code == "OTHER" or open_world_status == "needs_expert_review")),
        "feat_is_positive_review_uncertain_source": float(int(open_world_status == "positive_review_uncertain")),
    }
    return features


def zero_open_world_extra_features() -> dict[str, float]:
    return {field: 0.0 for field in OPEN_WORLD_EXTRA_FEATURE_COLUMNS}


def build_feature_row(
    row: dict[str, str],
    metadata_columns: list[str],
    consensus: dict[str, Any],
    include_open_world_extras: bool,
    role_map: dict[str, dict[str, str]],
) -> dict[str, Any]:
    output = {column: row.get(column, "") for column in metadata_columns}
    add_fold_roles(output, role_map)
    common_features = compute_common_features(row, consensus)
    extras = compute_open_world_extra_features(row, consensus) if include_open_world_extras else zero_open_world_extra_features()
    for feature_name in ALL_FEATURE_COLUMNS:
        feature_value = common_features.get(feature_name, extras.get(feature_name, 0.0))
        if feature_name in extras:
            feature_value = extras[feature_name]
        output[feature_name] = 0.0 if feature_value is None or (isinstance(feature_value, float) and math.isnan(feature_value)) else feature_value
    return output


def make_guard_v2(
    feature_columns: list[str],
    all_output_columns: list[str],
) -> dict[str, Any]:
    metadata_only_fields = sorted(
        field
        for field in all_output_columns
        if not field.startswith("feat_") and field not in SUPERVISION_COLUMNS
    )
    supervision_only_fields = sorted(field for field in all_output_columns if field in SUPERVISION_COLUMNS)
    forbidden_fields = sorted(set(FORBIDDEN_FEATURE_FIELDS + metadata_only_fields + supervision_only_fields))
    return {
        "allowed_feature_fields": feature_columns,
        "metadata_only_fields": metadata_only_fields,
        "supervision_only_fields": supervision_only_fields,
        "forbidden_feature_fields": forbidden_fields,
    }


def feature_summary_row(table_name: str, rows: list[dict[str, Any]], metadata_columns: list[str]) -> dict[str, Any]:
    zero_variance = 0
    if rows:
        for feature in ALL_FEATURE_COLUMNS:
            values = {float(row[feature]) for row in rows}
            if len(values) <= 1:
                zero_variance += 1
    return {
        "table_name": table_name,
        "row_count": len(rows),
        "feature_count": len(ALL_FEATURE_COLUMNS),
        "metadata_count": len(metadata_columns) + len(FOLD_COLUMNS),
        "missing_feature_cells": 0,
        "zero_variance_feature_count": zero_variance,
    }


def main() -> int:
    ensure_output_dir()

    seed_rows = read_csv(SEED_INPUT)
    open_world_rows = read_csv(OPEN_WORLD_INPUT)
    benchmark_rows = read_csv(BENCHMARK_INPUT)
    review_removed_rows = read_csv(REVIEW_REMOVED_INPUT)
    folds = json.loads(FOLDS_INPUT.read_text(encoding="utf-8"))

    role_map = build_fold_role_map(folds)
    seed_consensus = build_consensus_maps(seed_rows)
    open_world_consensus = build_consensus_maps(open_world_rows)
    challenge_rows = benchmark_rows + review_removed_rows
    challenge_consensus = build_consensus_maps(challenge_rows)

    seed_feature_rows = [
        build_feature_row(row, SEED_METADATA_COLUMNS, seed_consensus, include_open_world_extras=False, role_map=role_map)
        for row in seed_rows
    ]
    open_world_feature_rows = [
        build_feature_row(row, OPEN_WORLD_METADATA_COLUMNS, open_world_consensus, include_open_world_extras=True, role_map=role_map)
        for row in open_world_rows
    ]
    challenge_feature_rows = [
        build_feature_row(row, CHALLENGE_METADATA_COLUMNS, challenge_consensus, include_open_world_extras=False, role_map=role_map)
        for row in challenge_rows
    ]

    seed_fieldnames = SEED_METADATA_COLUMNS + FOLD_COLUMNS + ALL_FEATURE_COLUMNS
    open_world_fieldnames = OPEN_WORLD_METADATA_COLUMNS + FOLD_COLUMNS + ALL_FEATURE_COLUMNS
    challenge_fieldnames = CHALLENGE_METADATA_COLUMNS + FOLD_COLUMNS + ALL_FEATURE_COLUMNS

    write_csv(SEED_OUTPUT, seed_feature_rows, seed_fieldnames)
    write_csv(OPEN_WORLD_OUTPUT, open_world_feature_rows, open_world_fieldnames)
    write_csv(CHALLENGE_OUTPUT, challenge_feature_rows, challenge_fieldnames)

    FEATURE_COLUMNS_OUTPUT.write_text(json.dumps(ALL_FEATURE_COLUMNS, ensure_ascii=False, indent=2), encoding="utf-8")

    all_output_columns = sorted(set(seed_fieldnames + open_world_fieldnames + challenge_fieldnames))
    guard_v2 = make_guard_v2(ALL_FEATURE_COLUMNS, all_output_columns)
    GUARD_V2_OUTPUT.write_text(json.dumps(guard_v2, ensure_ascii=False, indent=2), encoding="utf-8")

    summary_rows = [
        feature_summary_row("seed_alignment_features.csv", seed_feature_rows, SEED_METADATA_COLUMNS),
        feature_summary_row("open_world_stability_features.csv", open_world_feature_rows, OPEN_WORLD_METADATA_COLUMNS),
        feature_summary_row("challenge_features.csv", challenge_feature_rows, CHALLENGE_METADATA_COLUMNS),
    ]
    write_csv(
        FEATURE_SUMMARY_OUTPUT,
        summary_rows,
        ["table_name", "row_count", "feature_count", "metadata_count", "missing_feature_cells", "zero_variance_feature_count"],
    )

    audit_lines = [
        "# RQ2 Feature Extraction Audit",
        "",
        f"- Generated at: {RUN_TS}",
        f"- Seed feature rows: {len(seed_feature_rows)}",
        f"- Open-world feature rows: {len(open_world_feature_rows)}",
        f"- Challenge feature rows: {len(challenge_feature_rows)}",
        f"- Feature column count: {len(ALL_FEATURE_COLUMNS)}",
        f"- All model-input fields use feat_ prefix: yes",
        "",
        "## Outputs",
        f"- {rel(SEED_OUTPUT)}",
        f"- {rel(OPEN_WORLD_OUTPUT)}",
        f"- {rel(CHALLENGE_OUTPUT)}",
        f"- {rel(FEATURE_COLUMNS_OUTPUT)}",
        f"- {rel(GUARD_V2_OUTPUT)}",
        f"- {rel(FEATURE_SUMMARY_OUTPUT)}",
    ]
    AUDIT_OUTPUT.write_text("\n".join(audit_lines) + "\n", encoding="utf-8")

    manifest = {
        "created_at_utc": RUN_TS,
        "build_script": rel(Path(__file__)),
        "input_files": {
            rel(SEED_INPUT): sha256_file(SEED_INPUT),
            rel(OPEN_WORLD_INPUT): sha256_file(OPEN_WORLD_INPUT),
            rel(BENCHMARK_INPUT): sha256_file(BENCHMARK_INPUT),
            rel(REVIEW_REMOVED_INPUT): sha256_file(REVIEW_REMOVED_INPUT),
            rel(FOLDS_INPUT): sha256_file(FOLDS_INPUT),
            rel(GUARD_INPUT): sha256_file(GUARD_INPUT),
        },
        "output_files": {},
        "counts": {
            "seed_feature_rows": len(seed_feature_rows),
            "open_world_feature_rows": len(open_world_feature_rows),
            "challenge_feature_rows": len(challenge_feature_rows),
            "feature_column_count": len(ALL_FEATURE_COLUMNS),
        },
    }
    for output_path in [
        SEED_OUTPUT,
        OPEN_WORLD_OUTPUT,
        CHALLENGE_OUTPUT,
        FEATURE_COLUMNS_OUTPUT,
        GUARD_V2_OUTPUT,
        FEATURE_SUMMARY_OUTPUT,
        AUDIT_OUTPUT,
    ]:
        manifest["output_files"][rel(output_path)] = sha256_file(output_path)
    MANIFEST_OUTPUT.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"seed_feature_rows={len(seed_feature_rows)}")
    print(f"open_world_feature_rows={len(open_world_feature_rows)}")
    print(f"challenge_feature_rows={len(challenge_feature_rows)}")
    print(f"feature_column_count={len(ALL_FEATURE_COLUMNS)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
