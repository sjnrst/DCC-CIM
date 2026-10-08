from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path.cwd().resolve()
FEATURE_DIR_V2 = ROOT / "artifacts" / "rq2_feature_tables_consensus_v2"
DATASET_DIR = ROOT / "artifacts" / "rq2_dual_track_dataset"
OUTPUT_DIR = ROOT / "artifacts" / "rq3_filtering_results"

SEED_FEATURES_CSV = FEATURE_DIR_V2 / "seed_alignment_features_consensus_v2.csv"
FEATURE_COLUMNS_JSON = FEATURE_DIR_V2 / "feature_columns_consensus_v2.json"
UNIVERSE_CSV = FEATURE_DIR_V2 / "unlabeled_candidate_universe.csv"
FOLDS_JSON = DATASET_DIR / "doc_group_folds_seed42.json"

OOF_SCORES_CSV = OUTPUT_DIR / "rq3_seed_oof_scores.csv"
SEED_METRICS_CSV = OUTPUT_DIR / "rq3_seed_filter_metrics.csv"
ERROR_REDUCTION_CSV = OUTPUT_DIR / "rq3_error_type_reduction.csv"
CHALLENGE_ROUTING_CSV = OUTPUT_DIR / "rq3_challenge_routing.csv"
CHALLENGE_SUMMARY_CSV = OUTPUT_DIR / "rq3_challenge_summary.csv"
OPEN_WORLD_ROUTING_CSV = OUTPUT_DIR / "rq3_open_world_routing.csv"
OPEN_WORLD_SUMMARY_CSV = OUTPUT_DIR / "rq3_open_world_summary.csv"
THRESHOLDS_CSV = OUTPUT_DIR / "rq3_thresholds_by_fold.csv"
METHOD_COMPARISON_CSV = OUTPUT_DIR / "rq3_method_comparison_summary.csv"
AUDIT_MD = OUTPUT_DIR / "rq3_filtering_audit.md"
VALIDATION_JSON = OUTPUT_DIR / "validation_results.json"
MANIFEST_JSON = OUTPUT_DIR / "run_manifest.json"

FORBIDDEN_FIELDS = {
    "candidate_key",
    "model_name",
    "doc_id",
    "pred_id",
    "matched_gt_id",
    "matched_gt_name",
    "score_final",
    "score_alignment",
    "lambda_value",
    "error_code",
    "candidate_final_label",
    "manual_review_final_label",
    "binary_label",
    "training_role",
    "source_pool",
    "analysis_branch",
    "open_world_status",
    "challenge_type",
}
CONSENSUS_FEATURES = {
    "feat_same_doc_exact_name_count",
    "feat_same_doc_similar_name_count",
    "feat_cross_model_exact_name_count",
    "feat_cross_model_same_parent_count",
    "feat_same_doc_same_parent_count",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def has_utf8_sig(path: Path) -> bool:
    with path.open("rb") as handle:
        return handle.read(3) == b"\xef\xbb\xbf"


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    seed_rows = read_csv(SEED_FEATURES_CSV)
    oof_rows = read_csv(OOF_SCORES_CSV)
    seed_metric_rows = read_csv(SEED_METRICS_CSV)
    error_rows = read_csv(ERROR_REDUCTION_CSV)
    challenge_rows = read_csv(CHALLENGE_ROUTING_CSV)
    open_world_rows = read_csv(OPEN_WORLD_ROUTING_CSV)
    threshold_rows = read_csv(THRESHOLDS_CSV)
    method_comparison_rows = read_csv(METHOD_COMPARISON_CSV)
    feature_columns = json.loads(FEATURE_COLUMNS_JSON.read_text(encoding="utf-8"))
    universe_rows = read_csv(UNIVERSE_CSV)
    folds = json.loads(FOLDS_JSON.read_text(encoding="utf-8"))
    manifest = json.loads(MANIFEST_JSON.read_text(encoding="utf-8"))

    method_names = sorted({row["method_name"] for row in oof_rows})
    seed_keys = {row["candidate_key"] for row in seed_rows}
    fold_by_doc = {}
    for fold in folds:
        for doc_id in fold["test_doc_ids"]:
            fold_by_doc[doc_id] = int(fold["fold_index"])

    oof_cover_ok = len({row["candidate_key"] for row in oof_rows}) == 347 and all(
        sum(1 for row in oof_rows if row["candidate_key"] == key) == len(method_names) for key in seed_keys
    )
    oof_test_fold_ok = all(fold_by_doc.get(row["doc_id"]) == int(row["fold_index"]) for row in oof_rows)
    selected_on_train_only_ok = all(row["selected_on_train_only"] == "true" for row in threshold_rows)

    no_filter_rows = [row for row in seed_metric_rows if row["method_name"] == "no_filter"]
    no_filter_correct = all(abs(float(row["recall_before"]) - 1.0) < 1e-12 for row in no_filter_rows)
    metrics_present = all(row["precision_after"] != "" and row["recall_after"] != "" and row["f1_after"] != "" for row in seed_metric_rows)
    error_codes_present = {row["error_code"] for row in error_rows}
    open_world_has_no_prf = set(open_world_rows[0].keys()).isdisjoint({"precision", "recall", "f1"}) if open_world_rows else True

    universe_models = {row["model_name"] for row in universe_rows}
    universe_docs = {row["doc_id"] for row in universe_rows}
    manifest_inputs_ok = all(
        manifest.get("input_files", {}).get(rel(path)) == sha256_file(path)
        for path in [SEED_FEATURES_CSV, FEATURE_COLUMNS_JSON, FOLDS_JSON, UNIVERSE_CSV]
    )
    output_paths = [
        OOF_SCORES_CSV,
        SEED_METRICS_CSV,
        ERROR_REDUCTION_CSV,
        CHALLENGE_ROUTING_CSV,
        CHALLENGE_SUMMARY_CSV,
        OPEN_WORLD_ROUTING_CSV,
        OPEN_WORLD_SUMMARY_CSV,
        THRESHOLDS_CSV,
        METHOD_COMPARISON_CSV,
        AUDIT_MD,
    ]
    manifest_outputs_ok = all(manifest.get("output_files", {}).get(rel(path)) == sha256_file(path) for path in output_paths)

    checks = {
        "seed_oof_covers_347_samples": oof_cover_ok,
        "each_seed_sample_has_prediction_for_each_method": oof_cover_ok,
        "each_fold_scores_only_test_docs": oof_test_fold_ok,
        "thresholds_selected_from_train_only": selected_on_train_only_ok,
        "no_forbidden_fields_in_training": FORBIDDEN_FIELDS.isdisjoint(set(feature_columns)),
        "binary_label_only_supervision": "binary_label" not in feature_columns,
        "error_code_only_error_analysis": "error_code" not in feature_columns,
        "no_filter_before_metrics_correct": no_filter_correct,
        "each_method_has_precision_recall_f1": metrics_present,
        "error_reduction_outputs_e1_to_e5": error_codes_present == {"E1", "E2", "E3", "E4", "E5"},
        "challenge_set_all_25_routed": len({row["candidate_key"] for row in challenge_rows}) == 25,
        "open_world_all_candidates_routed": len(open_world_rows) == 1976,
        "open_world_no_precision_recall_f1": open_world_has_no_prf,
        "consensus_v2_from_unlabeled_universe": len(universe_models) == 6 and len(universe_docs) == 50 and CONSENSUS_FEATURES.issubset(set(feature_columns)),
        "csv_utf8_sig": all(has_utf8_sig(path) for path in output_paths if path.suffix == ".csv"),
        "input_output_sha256_recorded": manifest_inputs_ok and manifest_outputs_ok,
    }

    results = {
        "validated_at_utc": datetime.now(timezone.utc).isoformat(),
        "checks": checks,
        "counts": {
            "method_count": len(method_names),
            "seed_oof_rows": len(oof_rows),
            "challenge_routing_rows": len(challenge_rows),
            "open_world_routing_rows": len(open_world_rows),
            "threshold_rows": len(threshold_rows),
            "method_comparison_rows": len(method_comparison_rows),
        },
        "ready_for_rq3_main_result": all(checks.values()),
    }
    VALIDATION_JSON.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    manifest["output_files"][rel(VALIDATION_JSON)] = sha256_file(VALIDATION_JSON)
    MANIFEST_JSON.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    for name, passed in checks.items():
        print(f"{name}={passed}")
    print(json.dumps(results["counts"], ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
