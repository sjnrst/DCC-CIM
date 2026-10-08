from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path.cwd().resolve()
INPUT_DIR = ROOT / "artifacts" / "rq2_dual_track_dataset"
OUTPUT_DIR = ROOT / "artifacts" / "rq2_feature_tables"

SEED_OUTPUT = OUTPUT_DIR / "seed_alignment_features.csv"
OPEN_WORLD_OUTPUT = OUTPUT_DIR / "open_world_stability_features.csv"
CHALLENGE_OUTPUT = OUTPUT_DIR / "challenge_features.csv"
FEATURE_COLUMNS_OUTPUT = OUTPUT_DIR / "feature_columns.json"
GUARD_V2_OUTPUT = OUTPUT_DIR / "feature_leakage_guard_v2.json"
FEATURE_SUMMARY_OUTPUT = OUTPUT_DIR / "feature_summary.csv"
AUDIT_OUTPUT = OUTPUT_DIR / "feature_extraction_audit.md"
VALIDATION_OUTPUT = OUTPUT_DIR / "validation_results.json"
MANIFEST_OUTPUT = OUTPUT_DIR / "run_manifest.json"
FOLDS_INPUT = INPUT_DIR / "doc_group_folds_seed42.json"

FORBIDDEN_FIELDS = {
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
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def has_utf8_sig(path: Path) -> bool:
    with path.open("rb") as handle:
        return handle.read(3) == b"\xef\xbb\xbf"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def can_float(value: Any) -> bool:
    text = "" if value is None else str(value).strip()
    if text == "":
        return False
    try:
        float(text)
        return True
    except ValueError:
        return False


def main() -> int:
    seed_rows = read_csv(SEED_OUTPUT)
    open_world_rows = read_csv(OPEN_WORLD_OUTPUT)
    challenge_rows = read_csv(CHALLENGE_OUTPUT)
    feature_columns = json.loads(FEATURE_COLUMNS_OUTPUT.read_text(encoding="utf-8"))
    guard_v2 = json.loads(GUARD_V2_OUTPUT.read_text(encoding="utf-8"))
    folds = json.loads(FOLDS_INPUT.read_text(encoding="utf-8"))
    manifest = json.loads(MANIFEST_OUTPUT.read_text(encoding="utf-8"))

    all_output_csvs = [SEED_OUTPUT, OPEN_WORLD_OUTPUT, CHALLENGE_OUTPUT, FEATURE_SUMMARY_OUTPUT]
    all_input_files = [
        INPUT_DIR / "seed_alignment_train_pool.csv",
        INPUT_DIR / "open_world_stability_pool.csv",
        INPUT_DIR / "benchmark_conflict_challenge_16.csv",
        INPUT_DIR / "positive_review_removed_challenge.csv",
        FOLDS_INPUT,
        INPUT_DIR / "feature_leakage_guard.json",
    ]
    all_output_files = all_output_csvs + [FEATURE_COLUMNS_OUTPUT, GUARD_V2_OUTPUT, AUDIT_OUTPUT]

    feat_columns_are_prefixed = all(column.startswith("feat_") for column in feature_columns)
    no_forbidden_in_feature_columns = set(feature_columns).isdisjoint(FORBIDDEN_FIELDS)

    non_numeric_feature_cells = 0
    missing_feature_cells = 0
    for rows in (seed_rows, open_world_rows, challenge_rows):
        for row in rows:
            for column in feature_columns:
                value = row.get(column, "")
                if str(value).strip() == "":
                    missing_feature_cells += 1
                elif not can_float(value):
                    non_numeric_feature_cells += 1

    fold_doc_roles: dict[int, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    fold_role_conflict = False
    for row in seed_rows:
        doc_id = row["doc_id"]
        for fold_index in range(5):
            role = row[f"fold_{fold_index}_role"]
            fold_doc_roles[fold_index][doc_id].add(role)
    for fold_index, doc_map in fold_doc_roles.items():
        for roles in doc_map.values():
            if len(roles) != 1:
                fold_role_conflict = True

    fold_pos_neg_ok = True
    for fold in folds:
        test_docs = set(fold["test_doc_ids"])
        train_docs = set(fold["train_doc_ids"])
        train_pos = train_neg = test_pos = test_neg = 0
        for row in seed_rows:
            doc_id = row["doc_id"]
            label = row["binary_label"]
            if doc_id in train_docs:
                if label == "1":
                    train_pos += 1
                elif label == "0":
                    train_neg += 1
            elif doc_id in test_docs:
                if label == "1":
                    test_pos += 1
                elif label == "0":
                    test_neg += 1
        if min(train_pos, train_neg, test_pos, test_neg) <= 0:
            fold_pos_neg_ok = False

    manifest_inputs_ok = all(manifest.get("input_files", {}).get(rel(path)) == sha256_file(path) for path in all_input_files)
    manifest_outputs_ok = all(manifest.get("output_files", {}).get(rel(path)) == sha256_file(path) for path in all_output_files)

    checks = {
        "seed_alignment_features_row_count_347": len(seed_rows) == 347,
        "open_world_stability_features_row_count_1976": len(open_world_rows) == 1976,
        "challenge_features_row_count_25": len(challenge_rows) == 25,
        "feature_columns_all_prefixed_feat": feat_columns_are_prefixed,
        "feature_columns_exclude_forbidden_fields": no_forbidden_in_feature_columns,
        "all_feat_fields_numeric": non_numeric_feature_cells == 0,
        "all_feat_fields_no_missing": missing_feature_cells == 0,
        "binary_label_not_in_feature_columns": "binary_label" not in feature_columns,
        "error_code_not_in_feature_columns": "error_code" not in feature_columns,
        "score_final_not_in_feature_columns": "score_final" not in feature_columns,
        "matched_gt_fields_not_in_feature_columns": "matched_gt_name" not in feature_columns and "matched_gt_id" not in feature_columns,
        "id_fields_not_in_feature_columns": all(field not in feature_columns for field in ("model_name", "doc_id", "pred_id", "candidate_key")),
        "fold_doc_role_consistent": not fold_role_conflict,
        "each_fold_train_and_test_have_pos_neg": fold_pos_neg_ok,
        "output_csv_utf8_sig": all(has_utf8_sig(path) for path in all_output_csvs),
        "input_output_sha256_recorded": manifest_inputs_ok and manifest_outputs_ok,
        "audit_report_generated": AUDIT_OUTPUT.exists(),
    }

    results = {
        "validated_at_utc": datetime.now(timezone.utc).isoformat(),
        "checks": checks,
        "counts": {
            "seed_rows": len(seed_rows),
            "open_world_rows": len(open_world_rows),
            "challenge_rows": len(challenge_rows),
            "feature_column_count": len(feature_columns),
            "missing_feature_cells": missing_feature_cells,
            "non_numeric_feature_cells": non_numeric_feature_cells,
        },
        "data_leakage_detected": not no_forbidden_in_feature_columns,
        "ready_for_model_training": all(checks.values()),
    }
    VALIDATION_OUTPUT.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    manifest["output_files"][rel(VALIDATION_OUTPUT)] = sha256_file(VALIDATION_OUTPUT)
    MANIFEST_OUTPUT.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    for name, passed in checks.items():
        print(f"{name}={passed}")
    print(json.dumps(results["counts"], ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
