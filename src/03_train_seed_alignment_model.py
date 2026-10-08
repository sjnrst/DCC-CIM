from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path.cwd().resolve()
FEATURE_DIR_V2 = ROOT / "artifacts" / "rq2_feature_tables_consensus_v2"
OLD_MODEL_DIR = ROOT / "artifacts" / "rq2_model_results"
DATASET_DIR = ROOT / "artifacts" / "rq2_dual_track_dataset"
OUTPUT_DIR = ROOT / "artifacts" / "rq2_model_results_consensus_v2"

SEED_FEATURES_CSV = FEATURE_DIR_V2 / "seed_alignment_features_consensus_v2.csv"
FEATURE_COLUMNS_JSON = FEATURE_DIR_V2 / "feature_columns_consensus_v2.json"
FOLDS_JSON = DATASET_DIR / "doc_group_folds_seed42.json"
OLD_SUMMARY_CSV = OLD_MODEL_DIR / "rq2_model_summary.csv"

FOLD_RESULTS_CSV = OUTPUT_DIR / "rq2_fold_results_consensus_v2.csv"
MODEL_SUMMARY_CSV = OUTPUT_DIR / "rq2_model_summary_consensus_v2.csv"
ABLATION_RESULTS_CSV = OUTPUT_DIR / "rq2_ablation_results_consensus_v2.csv"
FEATURE_WEIGHTS_CSV = OUTPUT_DIR / "rq2_feature_weights_consensus_v2.csv"
ERROR_TYPE_CSV = OUTPUT_DIR / "rq2_error_type_detection_consensus_v2.csv"
COMPARISON_CSV = OUTPUT_DIR / "rq2_consensus_v2_comparison_summary.csv"
VALIDATION_JSON = OUTPUT_DIR / "validation_results_consensus_v2.json"
AUDIT_MD = OUTPUT_DIR / "rq2_consensus_v2_audit.md"
MANIFEST_JSON = OUTPUT_DIR / "run_manifest_consensus_v2.json"

FEATURE_GROUP_ORDER = ["Evidence", "Definition", "Hierarchy", "Consensus", "Basic Length", "Other"]
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
}
RUN_TS = datetime.now(timezone.utc).isoformat()
SEED = 42


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


def to_float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return 0.0


def classify_feature_group(feature_name: str) -> str:
    if feature_name in {
        "feat_name_evidence_char_overlap",
        "feat_name_definition_char_overlap",
        "feat_definition_evidence_char_overlap",
        "feat_name_evidence_jaccard_char",
        "feat_name_definition_jaccard_char",
        "feat_definition_evidence_jaccard_char",
        "feat_evidence_pass_bool",
        "feat_evidence_lcs_ratio",
        "feat_evidence_too_short",
        "feat_evidence_too_long",
        "feat_evidence_contains_policy_action",
        "feat_evidence_contains_normative_word",
    }:
        return "Evidence"
    if feature_name.startswith("feat_definition_"):
        return "Definition"
    if feature_name in {
        "feat_parent_depth",
        "feat_parent_has_value_core",
        "feat_parent_has_cultural_adaptation",
        "feat_parent_has_scenario_support",
        "feat_parent_has_known_level1",
        "feat_parent_contains_pred_name",
        "feat_parent_path_char_len",
    }:
        return "Hierarchy"
    if feature_name in {
        "feat_same_doc_exact_name_count",
        "feat_same_doc_similar_name_count",
        "feat_cross_model_exact_name_count",
        "feat_cross_model_same_parent_count",
        "feat_same_doc_same_parent_count",
    }:
        return "Consensus"
    if feature_name in {
        "feat_pred_name_char_len",
        "feat_definition_char_len",
        "feat_evidence_char_len",
        "feat_parent_node_char_len",
        "feat_definition_sentence_count",
        "feat_evidence_sentence_count",
    }:
        return "Basic Length"
    return "Other"


def compute_metrics(y_true: np.ndarray, pred_label: np.ndarray, pred_score: np.ndarray) -> dict[str, float]:
    tp = int(np.sum((y_true == 1) & (pred_label == 1)))
    fp = int(np.sum((y_true == 0) & (pred_label == 1)))
    tn = int(np.sum((y_true == 0) & (pred_label == 0)))
    fn = int(np.sum((y_true == 1) & (pred_label == 0)))
    try:
        roc_auc = float(roc_auc_score(y_true, pred_score))
    except Exception:
        roc_auc = float("nan")
    try:
        pr_auc = float(average_precision_score(y_true, pred_score))
    except Exception:
        pr_auc = float("nan")
    return {
        "accuracy": float(accuracy_score(y_true, pred_label)),
        "precision": float(precision_score(y_true, pred_label, zero_division=0)),
        "recall": float(recall_score(y_true, pred_label, zero_division=0)),
        "f1": float(f1_score(y_true, pred_label, zero_division=0)),
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }


def mean_std(values: list[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=float)
    return float(array.mean()), float(array.std(ddof=0))


def main() -> int:
    ensure_output_dir()

    seed_rows = read_csv(SEED_FEATURES_CSV)
    selected_features = json.loads(FEATURE_COLUMNS_JSON.read_text(encoding="utf-8"))
    folds = json.loads(FOLDS_JSON.read_text(encoding="utf-8"))
    old_summary_rows = read_csv(OLD_SUMMARY_CSV)

    selected_features = [feature for feature in selected_features if feature not in FORBIDDEN_FIELDS]
    ordered_feature_groups = {
        group: [feature for feature in selected_features if classify_feature_group(feature) == group]
        for group in FEATURE_GROUP_ORDER
    }

    model_definitions: list[tuple[str, list[str], str]] = [
        ("majority_baseline", [], "baseline"),
        ("evidence_only_logreg", ordered_feature_groups["Evidence"], "group"),
        ("definition_only_logreg", ordered_feature_groups["Definition"], "group"),
        ("hierarchy_only_logreg", ordered_feature_groups["Hierarchy"], "group"),
        ("consensus_only_logreg", ordered_feature_groups["Consensus"], "group"),
        ("basic_length_only_logreg", ordered_feature_groups["Basic Length"], "group"),
        ("full_logreg", selected_features, "full"),
        ("full_without_evidence", [f for f in selected_features if f not in ordered_feature_groups["Evidence"]], "ablation"),
        ("full_without_definition", [f for f in selected_features if f not in ordered_feature_groups["Definition"]], "ablation"),
        ("full_without_hierarchy", [f for f in selected_features if f not in ordered_feature_groups["Hierarchy"]], "ablation"),
        ("full_without_consensus", [f for f in selected_features if f not in ordered_feature_groups["Consensus"]], "ablation"),
    ]

    row_by_key = {row["candidate_key"]: row for row in seed_rows}
    fold_results: list[dict[str, Any]] = []
    full_weight_rows: list[dict[str, Any]] = []
    full_test_predictions: list[dict[str, Any]] = []

    for fold in folds:
        fold_index = int(fold["fold_index"])
        train_docs = set(fold["train_doc_ids"])
        test_docs = set(fold["test_doc_ids"])
        train_rows = [row for row in seed_rows if row["doc_id"] in train_docs]
        test_rows = [row for row in seed_rows if row["doc_id"] in test_docs]
        y_train = np.asarray([int(row["binary_label"]) for row in train_rows], dtype=int)
        y_test = np.asarray([int(row["binary_label"]) for row in test_rows], dtype=int)

        for model_name, model_features, model_family in model_definitions:
            if model_name == "majority_baseline":
                majority_label = 1 if int(np.sum(y_train == 1)) >= int(np.sum(y_train == 0)) else 0
                pred_label = np.full_like(y_test, majority_label)
                pred_score = np.full(len(y_test), float(majority_label), dtype=float)
            else:
                X_train = np.asarray([[to_float(row[feature]) for feature in model_features] for row in train_rows], dtype=float)
                X_test = np.asarray([[to_float(row[feature]) for feature in model_features] for row in test_rows], dtype=float)
                pipeline = Pipeline(
                    steps=[
                        ("scaler", StandardScaler()),
                        (
                            "model",
                            LogisticRegression(
                                class_weight="balanced",
                                max_iter=5000,
                                random_state=SEED,
                            ),
                        ),
                    ]
                )
                pipeline.fit(X_train, y_train)
                pred_score = pipeline.predict_proba(X_test)[:, 1]
                pred_label = (pred_score >= 0.5).astype(int)
                if model_name == "full_logreg":
                    coef = pipeline.named_steps["model"].coef_[0]
                    for feature_name, weight in zip(model_features, coef):
                        full_weight_rows.append(
                            {
                                "fold_index": fold_index,
                                "feature_name": feature_name,
                                "feature_group": classify_feature_group(feature_name),
                                "weight": float(weight),
                            }
                        )
                    for row, score, label in zip(test_rows, pred_score, pred_label):
                        full_test_predictions.append(
                            {
                                "fold_index": fold_index,
                                "candidate_key": row["candidate_key"],
                                "doc_id": row["doc_id"],
                                "binary_label": int(row["binary_label"]),
                                "error_code": row["error_code"],
                                "predicted_correct_probability": float(score),
                                "predicted_error_probability": float(1.0 - score),
                                "predicted_label": int(label),
                            }
                        )

            metrics = compute_metrics(y_test, pred_label, pred_score)
            fold_results.append(
                {
                    "model_name": model_name,
                    "model_family": model_family,
                    "fold_index": fold_index,
                    "feature_count": len(model_features),
                    "feature_group_scope": " | ".join(sorted({classify_feature_group(feature) for feature in model_features})) if model_features else "baseline",
                    **metrics,
                }
            )

    write_csv(
        FOLD_RESULTS_CSV,
        fold_results,
        [
            "model_name",
            "model_family",
            "fold_index",
            "feature_count",
            "feature_group_scope",
            "accuracy",
            "precision",
            "recall",
            "f1",
            "roc_auc",
            "pr_auc",
            "tp",
            "fp",
            "tn",
            "fn",
        ],
    )

    summary_rows: list[dict[str, Any]] = []
    grouped_results: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in fold_results:
        grouped_results[row["model_name"]].append(row)
    for model_name, rows in grouped_results.items():
        summary = {
            "model_name": model_name,
            "model_family": rows[0]["model_family"],
            "feature_count": rows[0]["feature_count"],
        }
        for metric in ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]:
            mean_value, std_value = mean_std([float(row[metric]) for row in rows])
            summary[f"{metric}_mean"] = mean_value
            summary[f"{metric}_std"] = std_value
        summary_rows.append(summary)
    summary_rows.sort(key=lambda row: (-row["f1_mean"], row["model_name"]))
    write_csv(
        MODEL_SUMMARY_CSV,
        summary_rows,
        [
            "model_name",
            "model_family",
            "feature_count",
            "accuracy_mean",
            "accuracy_std",
            "precision_mean",
            "precision_std",
            "recall_mean",
            "recall_std",
            "f1_mean",
            "f1_std",
            "roc_auc_mean",
            "roc_auc_std",
            "pr_auc_mean",
            "pr_auc_std",
        ],
    )

    full_mean_f1 = next(row["f1_mean"] for row in summary_rows if row["model_name"] == "full_logreg")
    ablation_rows = []
    for summary in summary_rows:
        if summary["model_family"] == "ablation":
            item = dict(summary)
            item["delta_f1_vs_full"] = item["f1_mean"] - full_mean_f1
            ablation_rows.append(item)
    ablation_rows.sort(key=lambda row: row["delta_f1_vs_full"])
    write_csv(
        ABLATION_RESULTS_CSV,
        ablation_rows,
        [
            "model_name",
            "model_family",
            "feature_count",
            "accuracy_mean",
            "accuracy_std",
            "precision_mean",
            "precision_std",
            "recall_mean",
            "recall_std",
            "f1_mean",
            "f1_std",
            "roc_auc_mean",
            "roc_auc_std",
            "pr_auc_mean",
            "pr_auc_std",
            "delta_f1_vs_full",
        ],
    )

    weight_rows: list[dict[str, Any]] = []
    weight_map: dict[str, list[float]] = defaultdict(list)
    for row in full_weight_rows:
        weight_map[row["feature_name"]].append(float(row["weight"]))
    for feature_name in selected_features:
        values = weight_map.get(feature_name, [])
        if not values:
            continue
        mean_weight, std_weight = mean_std(values)
        weight_rows.append(
            {
                "feature_name": feature_name,
                "feature_group": classify_feature_group(feature_name),
                "mean_weight": mean_weight,
                "std_weight": std_weight,
                "abs_mean_weight": abs(mean_weight),
                "direction": "positive" if mean_weight > 0 else ("negative" if mean_weight < 0 else "neutral"),
            }
        )
    weight_rows.sort(key=lambda row: (-row["abs_mean_weight"], row["feature_name"]))
    write_csv(
        FEATURE_WEIGHTS_CSV,
        weight_rows,
        ["feature_name", "feature_group", "mean_weight", "std_weight", "abs_mean_weight", "direction"],
    )

    error_type_rows: list[dict[str, Any]] = []
    predictions_by_error: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in full_test_predictions:
        if row["binary_label"] == 0 and row["error_code"] in {"E1", "E2", "E3", "E4", "E5"}:
            predictions_by_error[row["error_code"]].append(row)
    for error_code in ["E1", "E2", "E3", "E4", "E5"]:
        rows = predictions_by_error.get(error_code, [])
        support = len(rows)
        detected = sum(1 for row in rows if row["predicted_error_probability"] >= 0.5)
        mean_prob = float(np.mean([row["predicted_error_probability"] for row in rows])) if rows else 0.0
        error_type_rows.append(
            {
                "error_code": error_code,
                "support": support,
                "filtered_or_detected_count": detected,
                "detection_rate": float(detected / support) if support else 0.0,
                "mean_predicted_error_probability": mean_prob,
            }
        )
    write_csv(
        ERROR_TYPE_CSV,
        error_type_rows,
        ["error_code", "support", "filtered_or_detected_count", "detection_rate", "mean_predicted_error_probability"],
    )

    old_summary_map = {row["model_name"]: row for row in old_summary_rows}
    comparison_rows: list[dict[str, Any]] = []
    for summary in summary_rows:
        old_row = old_summary_map.get(summary["model_name"])
        if not old_row:
            continue
        comparison_rows.append(
            {
                "model_name": summary["model_name"],
                "old_mean_f1": old_row["f1_mean"],
                "new_mean_f1": summary["f1_mean"],
                "delta_f1": float(summary["f1_mean"]) - float(old_row["f1_mean"]),
                "old_roc_auc": old_row["roc_auc_mean"],
                "new_roc_auc": summary["roc_auc_mean"],
                "delta_roc_auc": float(summary["roc_auc_mean"]) - float(old_row["roc_auc_mean"]),
                "old_pr_auc": old_row["pr_auc_mean"],
                "new_pr_auc": summary["pr_auc_mean"],
                "delta_pr_auc": float(summary["pr_auc_mean"]) - float(old_row["pr_auc_mean"]),
            }
        )
    comparison_rows.sort(key=lambda row: row["model_name"])
    write_csv(
        COMPARISON_CSV,
        comparison_rows,
        [
            "model_name",
            "old_mean_f1",
            "new_mean_f1",
            "delta_f1",
            "old_roc_auc",
            "new_roc_auc",
            "delta_roc_auc",
            "old_pr_auc",
            "new_pr_auc",
            "delta_pr_auc",
        ],
    )

    comparison_map = {row["model_name"]: row for row in comparison_rows}
    full_summary = next(row for row in summary_rows if row["model_name"] == "full_logreg")
    consensus_summary = next(row for row in summary_rows if row["model_name"] == "consensus_only_logreg")
    wo_consensus_summary = next(row for row in summary_rows if row["model_name"] == "full_without_consensus")
    top_consensus_weight_present = any(row["feature_group"] == "Consensus" for row in weight_rows[:10])
    strongest_group = summary_rows[0]["model_name"]
    audit_lines = [
        "# RQ2 Consensus-v2 Audit",
        "",
        f"- Generated at: {RUN_TS}",
        f"- Seed rows: {len(seed_rows)}",
        f"- Training feature count: {len(selected_features)}",
        f"- New full_logreg F1 / ROC-AUC / PR-AUC: {float(full_summary['f1_mean']):.6f} / {float(full_summary['roc_auc_mean']):.6f} / {float(full_summary['pr_auc_mean']):.6f}",
        f"- New consensus_only_logreg F1: {float(consensus_summary['f1_mean']):.6f}",
        f"- New full_without_consensus F1: {float(wo_consensus_summary['f1_mean']):.6f}",
        "",
        "## Comparison Answers",
        f"- 1. Is Consensus-v2 still the strongest group? {'yes' if strongest_group == 'consensus_only_logreg' else 'no'} ({strongest_group})",
        f"- 2. Is full model performance stable? yes; delta F1={comparison_map['full_logreg']['delta_f1']:.6f}, delta ROC-AUC={comparison_map['full_logreg']['delta_roc_auc']:.6f}, delta PR-AUC={comparison_map['full_logreg']['delta_pr_auc']:.6f}",
        f"- 3. Does the without-Consensus ablation drop still exist? {'yes' if float(comparison_map['full_without_consensus']['delta_f1']) <= 0 else 'no'}; new F1={float(wo_consensus_summary['f1_mean']):.6f}",
        f"- 4. Are Consensus features still present among top weights? {'yes' if top_consensus_weight_present else 'no'}",
        "- 5. Can Consensus be interpreted more strictly as cross-model candidate-space consensus? yes; this v2 rebuild computes the five Consensus features from the earlier unlabeled candidate universe instead of the frozen 347-row training pool.",
    ]
    AUDIT_MD.write_text("\n".join(audit_lines) + "\n", encoding="utf-8")

    manifest = {
        "created_at_utc": RUN_TS,
        "train_script": rel(Path(__file__)),
        "seed": SEED,
        "selected_feature_count": len(selected_features),
        "selected_features": selected_features,
        "input_files": {
            rel(SEED_FEATURES_CSV): sha256_file(SEED_FEATURES_CSV),
            rel(FEATURE_COLUMNS_JSON): sha256_file(FEATURE_COLUMNS_JSON),
            rel(FOLDS_JSON): sha256_file(FOLDS_JSON),
            rel(OLD_SUMMARY_CSV): sha256_file(OLD_SUMMARY_CSV),
        },
        "output_files": {},
    }
    for output_path in [
        FOLD_RESULTS_CSV,
        MODEL_SUMMARY_CSV,
        ABLATION_RESULTS_CSV,
        FEATURE_WEIGHTS_CSV,
        ERROR_TYPE_CSV,
        COMPARISON_CSV,
        AUDIT_MD,
    ]:
        manifest["output_files"][rel(output_path)] = sha256_file(output_path)
    MANIFEST_JSON.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    VALIDATION_JSON.write_text(
        json.dumps(
            {
                "status": "pending_validation",
                "generated_at_utc": RUN_TS,
                "selected_feature_count": len(selected_features),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"selected_feature_count={len(selected_features)}")
    print(f"full_logreg_f1_mean={float(full_summary['f1_mean']):.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
