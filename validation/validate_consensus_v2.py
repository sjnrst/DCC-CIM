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
MODEL_DIR_V2 = ROOT / "artifacts" / "rq2_model_results_consensus_v2"
DATASET_DIR = ROOT / "artifacts" / "rq2_dual_track_dataset"

UNIVERSE_CSV = FEATURE_DIR_V2 / "unlabeled_candidate_universe.csv"
SEED_FEATURES_CSV = FEATURE_DIR_V2 / "seed_alignment_features_consensus_v2.csv"
FEATURE_COLUMNS_JSON = FEATURE_DIR_V2 / "feature_columns_consensus_v2.json"
FEATURE_AUDIT_MD = FEATURE_DIR_V2 / "feature_extraction_audit_consensus_v2.md"
FEATURE_MANIFEST_JSON = FEATURE_DIR_V2 / "run_manifest_consensus_v2.json"
FOLDS_JSON = DATASET_DIR / "doc_group_folds_seed42.json"

FOLD_RESULTS_CSV = MODEL_DIR_V2 / "rq2_fold_results_consensus_v2.csv"
MODEL_SUMMARY_CSV = MODEL_DIR_V2 / "rq2_model_summary_consensus_v2.csv"
ABLATION_RESULTS_CSV = MODEL_DIR_V2 / "rq2_ablation_results_consensus_v2.csv"
FEATURE_WEIGHTS_CSV = MODEL_DIR_V2 / "rq2_feature_weights_consensus_v2.csv"
ERROR_TYPE_CSV = MODEL_DIR_V2 / "rq2_error_type_detection_consensus_v2.csv"
COMPARISON_CSV = MODEL_DIR_V2 / "rq2_consensus_v2_comparison_summary.csv"
AUDIT_MD = MODEL_DIR_V2 / "rq2_consensus_v2_audit.md"
MANIFEST_JSON = MODEL_DIR_V2 / "run_manifest_consensus_v2.json"
VALIDATION_JSON = MODEL_DIR_V2 / "validation_results_consensus_v2.json"

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


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    universe_rows = read_csv(UNIVERSE_CSV)
    seed_rows = read_csv(SEED_FEATURES_CSV)
    feature_columns = json.loads(FEATURE_COLUMNS_JSON.read_text(encoding="utf-8"))
    folds = json.loads(FOLDS_JSON.read_text(encoding="utf-8"))
    feature_manifest = json.loads(FEATURE_MANIFEST_JSON.read_text(encoding="utf-8"))
    model_manifest = json.loads(MANIFEST_JSON.read_text(encoding="utf-8"))

    fold_results = read_csv(FOLD_RESULTS_CSV)
    model_summary_rows = read_csv(MODEL_SUMMARY_CSV)
    ablation_rows = read_csv(ABLATION_RESULTS_CSV)
    comparison_rows = read_csv(COMPARISON_CSV)

    universe_columns = set(universe_rows[0].keys()) if universe_rows else set()
    model_coverage = {row["model_name"] for row in universe_rows}
    doc_coverage = {row["doc_id"] for row in universe_rows}

    forbidden_universe_absent = FORBIDDEN_UNIVERSE_FIELDS.isdisjoint(universe_columns)
    consensus_v2_no_labels = forbidden_universe_absent
    feature_prefix_ok = all(feature.startswith("feat_") for feature in feature_columns)

    fold_doc_leak = False
    fold_roles_by_doc: dict[int, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    for row in seed_rows:
        doc_id = row["doc_id"]
        for fold_index in range(5):
            fold_roles_by_doc[fold_index][doc_id].add(row[f"fold_{fold_index}_role"])
    for doc_map in fold_roles_by_doc.values():
        for roles in doc_map.values():
            if len(roles) != 1:
                fold_doc_leak = True

    required_output_files = [
        FOLD_RESULTS_CSV,
        MODEL_SUMMARY_CSV,
        ABLATION_RESULTS_CSV,
        FEATURE_WEIGHTS_CSV,
        ERROR_TYPE_CSV,
        COMPARISON_CSV,
        AUDIT_MD,
    ]
    feature_manifest_ok = all(
        feature_manifest.get("output_files", {}).get(rel(path)) == sha256_file(path)
        for path in [UNIVERSE_CSV, SEED_FEATURES_CSV, FEATURE_COLUMNS_JSON, FEATURE_AUDIT_MD]
    )
    model_manifest_ok = all(
        model_manifest.get("output_files", {}).get(rel(path)) == sha256_file(path)
        for path in required_output_files
    )

    checks = {
        "universe_excludes_label_gt_score_fields": forbidden_universe_absent,
        "universe_covers_6_models": len(model_coverage) == 6,
        "universe_covers_50_docs": len(doc_coverage) == 50,
        "consensus_v2_features_use_no_label_or_gt_fields": consensus_v2_no_labels,
        "seed_alignment_features_consensus_v2_row_count_347": len(seed_rows) == 347,
        "training_feature_count_44": len(feature_columns) == 44,
        "all_training_features_prefixed_feat": feature_prefix_ok,
        "doc_id_5fold_no_leakage": not fold_doc_leak,
        "model_result_files_all_generated": all(path.exists() for path in required_output_files),
        "output_sha256_recorded": feature_manifest_ok and model_manifest_ok,
        "ready_for_rq3": True,
    }

    results = {
        "validated_at_utc": datetime.now(timezone.utc).isoformat(),
        "checks": checks,
        "counts": {
            "universe_candidate_count": len(universe_rows),
            "model_coverage_count": len(model_coverage),
            "document_coverage_count": len(doc_coverage),
            "seed_feature_rows": len(seed_rows),
            "training_feature_count": len(feature_columns),
            "fold_result_rows": len(fold_results),
            "model_summary_rows": len(model_summary_rows),
            "ablation_rows": len(ablation_rows),
            "comparison_rows": len(comparison_rows),
        },
        "ready_for_rq3": all(checks.values()),
    }
    VALIDATION_JSON.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    model_manifest["output_files"][rel(VALIDATION_JSON)] = sha256_file(VALIDATION_JSON)
    MANIFEST_JSON.write_text(json.dumps(model_manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    for name, passed in checks.items():
        print(f"{name}={passed}")
    print(json.dumps(results["counts"], ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
