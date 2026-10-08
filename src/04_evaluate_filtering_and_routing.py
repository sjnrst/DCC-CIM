from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
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
FEATURE_DIR_OLD = ROOT / "artifacts" / "rq2_feature_tables"
DATASET_DIR = ROOT / "artifacts" / "rq2_dual_track_dataset"
RQ2_MODEL_DIR = ROOT / "artifacts" / "rq2_model_results_consensus_v2"
OUTPUT_DIR = ROOT / "artifacts" / "rq3_filtering_results"

SEED_FEATURES_CSV = FEATURE_DIR_V2 / "seed_alignment_features_consensus_v2.csv"
FEATURE_COLUMNS_JSON = FEATURE_DIR_V2 / "feature_columns_consensus_v2.json"
UNIVERSE_CSV = FEATURE_DIR_V2 / "unlabeled_candidate_universe.csv"
FOLDS_JSON = DATASET_DIR / "doc_group_folds_seed42.json"
CHALLENGE_FEATURES_CSV = FEATURE_DIR_OLD / "challenge_features.csv"
OPEN_WORLD_FEATURES_CSV = FEATURE_DIR_OLD / "open_world_stability_features.csv"
OPEN_WORLD_POOL_CSV = DATASET_DIR / "open_world_stability_pool.csv"
RQ2_SUMMARY_CSV = RQ2_MODEL_DIR / "rq2_model_summary_consensus_v2.csv"
RQ2_ABLATION_CSV = RQ2_MODEL_DIR / "rq2_ablation_results_consensus_v2.csv"

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

CONSENSUS_FEATURES = [
    "feat_same_doc_exact_name_count",
    "feat_same_doc_similar_name_count",
    "feat_cross_model_exact_name_count",
    "feat_cross_model_same_parent_count",
    "feat_same_doc_same_parent_count",
]
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
GRID = [round(value, 2) for value in np.arange(0.05, 0.951, 0.01)]
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


def cleaned_char_set(text: str) -> set[str]:
    text = "" if text is None else str(text).strip()
    return {
        char
        for char in text
        if not char.isspace() and char not in "，。！？；;,.、“”\"'：:（）()[]【】/|_-"
    }


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
    if feature_name in set(CONSENSUS_FEATURES):
        return "Consensus-v2"
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


def build_consensus_maps(universe_rows: list[dict[str, str]]) -> dict[str, Any]:
    same_doc_exact_name = Counter((row["doc_id"], row["pred_name"]) for row in universe_rows)
    same_doc_same_parent = Counter((row["doc_id"], row["pred_parent_node"]) for row in universe_rows)
    doc_name_model_sets: dict[tuple[str, str], set[str]] = defaultdict(set)
    doc_parent_model_sets: dict[tuple[str, str], set[str]] = defaultdict(set)
    rows_by_doc: dict[str, list[dict[str, str]]] = defaultdict(list)
    similar_name_count: dict[str, int] = {}

    for row in universe_rows:
        doc_name_model_sets[(row["doc_id"], row["pred_name"])].add(row["model_name"])
        doc_parent_model_sets[(row["doc_id"], row["pred_parent_node"])].add(row["model_name"])
        rows_by_doc[row["doc_id"]].append(row)

    for doc_rows in rows_by_doc.values():
        char_sets = {row["candidate_key"]: cleaned_char_set(row["pred_name"]) for row in doc_rows}
        for row in doc_rows:
            base = char_sets[row["candidate_key"]]
            count = 0
            for other in doc_rows:
                other_set = char_sets[other["candidate_key"]]
                union = base | other_set
                score = 0.0 if not union else len(base & other_set) / len(union)
                if score >= 0.6:
                    count += 1
            similar_name_count[row["candidate_key"]] = count

    return {
        "same_doc_exact_name": same_doc_exact_name,
        "same_doc_same_parent": same_doc_same_parent,
        "cross_model_exact_name": {key: len(value) for key, value in doc_name_model_sets.items()},
        "cross_model_same_parent": {key: len(value) for key, value in doc_parent_model_sets.items()},
        "similar_name_count": similar_name_count,
    }


def replace_consensus_features(rows: list[dict[str, str]], consensus: dict[str, Any]) -> list[dict[str, str]]:
    updated_rows: list[dict[str, str]] = []
    for row in rows:
        updated = dict(row)
        key = f"{row['model_name']}|{row['doc_id']}|{row['pred_id']}"
        updated["feat_same_doc_exact_name_count"] = str(float(consensus["same_doc_exact_name"][(row["doc_id"], row["pred_name"])]))
        updated["feat_same_doc_similar_name_count"] = str(float(consensus["similar_name_count"].get(key, 1)))
        updated["feat_cross_model_exact_name_count"] = str(float(consensus["cross_model_exact_name"].get((row["doc_id"], row["pred_name"]), 0)))
        updated["feat_cross_model_same_parent_count"] = str(float(consensus["cross_model_same_parent"].get((row["doc_id"], row["pred_parent_node"]), 0)))
        updated["feat_same_doc_same_parent_count"] = str(float(consensus["same_doc_same_parent"][(row["doc_id"], row["pred_parent_node"])]))
        updated_rows.append(updated)
    return updated_rows


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
        "precision": float(precision_score(y_true, pred_label, zero_division=0)),
        "recall": float(recall_score(y_true, pred_label, zero_division=0)),
        "f1": float(f1_score(y_true, pred_label, zero_division=0)),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
    }


def mean_std(values: list[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=float)
    return float(array.mean()), float(array.std(ddof=0))


def find_best_f1_threshold(y_true: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    best_threshold = GRID[0]
    best_f1 = -1.0
    for threshold in GRID:
        pred = (scores >= threshold).astype(int)
        score = float(f1_score(y_true, pred, zero_division=0))
        if score > best_f1 or (abs(score - best_f1) < 1e-12 and threshold > best_threshold):
            best_threshold = threshold
            best_f1 = score
    return best_threshold, best_f1


def choose_binary_threshold(y_true: np.ndarray, scores: np.ndarray) -> tuple[float, str]:
    positive_mask = y_true == 1
    valid = []
    for threshold in GRID:
        kept = scores >= threshold
        recall = float(np.sum(kept & positive_mask) / np.sum(positive_mask))
        if recall >= 0.90:
            valid.append(threshold)
    if valid:
        return max(valid), "max_threshold_with_train_positive_recall_ge_0.90"
    best_threshold, _ = find_best_f1_threshold(y_true, scores)
    return best_threshold, "fallback_train_f1_max"


def choose_three_way_thresholds(y_true: np.ndarray, scores: np.ndarray) -> tuple[float, float, str]:
    positive_mask = y_true == 1
    tau_filter_candidates = []
    for threshold in GRID:
        not_filtered = scores >= threshold
        recall = float(np.sum(not_filtered & positive_mask) / np.sum(positive_mask))
        if recall >= 0.95:
            tau_filter_candidates.append(threshold)
    if tau_filter_candidates:
        tau_filter = max(tau_filter_candidates)
        note_filter = "max_tau_filter_with_positive_recall_ge_0.95"
    else:
        tau_filter, _ = find_best_f1_threshold(y_true, scores)
        note_filter = "fallback_tau_filter_train_f1_max"

    tau_keep_candidates = []
    for threshold in GRID:
        keep_mask = scores >= threshold
        keep_count = int(np.sum(keep_mask))
        if keep_count == 0:
            continue
        precision = float(np.sum((y_true == 1) & keep_mask) / keep_count)
        if precision >= 0.80:
            tau_keep_candidates.append((threshold, keep_count))
    if tau_keep_candidates:
        max_keep_count = max(item[1] for item in tau_keep_candidates)
        eligible = [threshold for threshold, keep_count in tau_keep_candidates if keep_count == max_keep_count]
        tau_keep = min(eligible)
        note_keep = "min_tau_keep_with_precision_ge_0.80_and_max_keep_count"
    else:
        tau_keep, _ = find_best_f1_threshold(y_true, scores)
        note_keep = "fallback_tau_keep_train_f1_max"

    note_adjust = ""
    if tau_filter >= tau_keep:
        smaller_valid = [threshold for threshold in GRID if threshold < tau_keep and threshold in tau_filter_candidates]
        if smaller_valid:
            tau_filter = max(smaller_valid)
            note_adjust = "adjusted_tau_filter_to_be_below_tau_keep"
        else:
            next_higher = next((threshold for threshold in GRID if threshold > tau_filter), None)
            if next_higher is not None:
                tau_keep = next_higher
                note_adjust = "adjusted_tau_keep_to_be_above_tau_filter"
            else:
                tau_filter = max(0.05, round(tau_keep - 0.01, 2))
                note_adjust = "forced_tau_filter_below_tau_keep"
    note = "; ".join(part for part in [note_filter, note_keep, note_adjust] if part)
    return tau_filter, tau_keep, note


def get_feature_groups(selected_features: list[str]) -> dict[str, list[str]]:
    return {
        "Evidence": [feature for feature in selected_features if classify_feature_group(feature) == "Evidence"],
        "Consensus-v2": [feature for feature in selected_features if classify_feature_group(feature) == "Consensus-v2"],
        "Full Diagnostic-v2": list(selected_features),
        "Full without Evidence-v2": [feature for feature in selected_features if classify_feature_group(feature) != "Evidence"],
    }


def normalize_minmax(values: list[float]) -> list[float]:
    if not values:
        return []
    min_value = min(values)
    max_value = max(values)
    if abs(max_value - min_value) < 1e-12:
        return [0.0 for _ in values]
    return [(value - min_value) / (max_value - min_value) for value in values]


def main() -> int:
    ensure_output_dir()

    seed_rows = read_csv(SEED_FEATURES_CSV)
    selected_features = [feature for feature in json.loads(FEATURE_COLUMNS_JSON.read_text(encoding="utf-8")) if feature not in FORBIDDEN_FIELDS]
    universe_rows = read_csv(UNIVERSE_CSV)
    challenge_rows = replace_consensus_features(read_csv(CHALLENGE_FEATURES_CSV), build_consensus_maps(universe_rows))
    open_world_rows = replace_consensus_features(read_csv(OPEN_WORLD_FEATURES_CSV), build_consensus_maps(universe_rows))
    folds = json.loads(FOLDS_JSON.read_text(encoding="utf-8"))
    rq2_summary_rows = read_csv(RQ2_SUMMARY_CSV)
    rq2_ablation_rows = read_csv(RQ2_ABLATION_CSV)

    feature_groups = get_feature_groups(selected_features)
    method_definitions: list[tuple[str, list[str], str]] = [
        ("no_filter", [], "baseline"),
        ("evidence_only_logreg", feature_groups["Evidence"], "Evidence"),
        ("consensus_v2_only_logreg", feature_groups["Consensus-v2"], "Consensus-v2"),
        ("full_diagnostic_v2_logreg", feature_groups["Full Diagnostic-v2"], "Full Diagnostic-v2"),
        ("full_without_evidence_v2_logreg", feature_groups["Full without Evidence-v2"], "Full without Evidence-v2"),
    ]

    row_by_key = {row["candidate_key"]: row for row in seed_rows}
    oof_rows: list[dict[str, Any]] = []
    threshold_rows: list[dict[str, Any]] = []
    seed_method_predictions: dict[str, list[dict[str, Any]]] = defaultdict(list)
    challenge_routing_rows: list[dict[str, Any]] = []
    trained_models: dict[tuple[str, int], Pipeline | None] = {}

    for fold in folds:
        fold_index = int(fold["fold_index"])
        train_docs = set(fold["train_doc_ids"])
        test_docs = set(fold["test_doc_ids"])
        train_rows = [row for row in seed_rows if row["doc_id"] in train_docs]
        test_rows = [row for row in seed_rows if row["doc_id"] in test_docs]
        y_train = np.asarray([int(row["binary_label"]) for row in train_rows], dtype=int)

        for method_name, method_features, scope in method_definitions:
            if method_name == "no_filter":
                binary_threshold = 0.0
                tau_filter = 0.0
                tau_keep = 0.0
                threshold_note = "no_filter_keep_all"
                trained_models[(method_name, fold_index)] = None
                for row in test_rows:
                    record = {
                        "candidate_key": row["candidate_key"],
                        "doc_id": row["doc_id"],
                        "fold_index": fold_index,
                        "binary_label": row["binary_label"],
                        "error_code": row["error_code"],
                        "method_name": method_name,
                        "p_correct": 1.0,
                        "p_error": 0.0,
                        "binary_threshold": binary_threshold,
                        "binary_decision": "keep",
                        "tau_filter": tau_filter,
                        "tau_keep": tau_keep,
                        "three_way_decision": "keep",
                    }
                    oof_rows.append(record)
                    seed_method_predictions[method_name].append(record)
            else:
                X_train = np.asarray([[to_float(row[feature]) for feature in method_features] for row in train_rows], dtype=float)
                X_test = np.asarray([[to_float(row[feature]) for feature in method_features] for row in test_rows], dtype=float)
                pipeline = Pipeline(
                    steps=[
                        ("scaler", StandardScaler()),
                        ("model", LogisticRegression(class_weight="balanced", max_iter=5000, random_state=SEED)),
                    ]
                )
                pipeline.fit(X_train, y_train)
                trained_models[(method_name, fold_index)] = pipeline
                train_scores = pipeline.predict_proba(X_train)[:, 1]
                test_scores = pipeline.predict_proba(X_test)[:, 1]
                binary_threshold, binary_note = choose_binary_threshold(y_train, train_scores)
                tau_filter, tau_keep, three_way_note = choose_three_way_thresholds(y_train, train_scores)
                threshold_note = f"binary={binary_note}; three_way={three_way_note}"

                for row, score in zip(test_rows, test_scores):
                    binary_decision = "keep" if score >= binary_threshold else "filter"
                    if score < tau_filter:
                        three_way_decision = "filter"
                    elif score < tau_keep:
                        three_way_decision = "review"
                    else:
                        three_way_decision = "keep"
                    record = {
                        "candidate_key": row["candidate_key"],
                        "doc_id": row["doc_id"],
                        "fold_index": fold_index,
                        "binary_label": row["binary_label"],
                        "error_code": row["error_code"],
                        "method_name": method_name,
                        "p_correct": float(score),
                        "p_error": float(1.0 - score),
                        "binary_threshold": binary_threshold,
                        "binary_decision": binary_decision,
                        "tau_filter": tau_filter,
                        "tau_keep": tau_keep,
                        "three_way_decision": three_way_decision,
                    }
                    oof_rows.append(record)
                    seed_method_predictions[method_name].append(record)

            threshold_rows.append(
                {
                    "method_name": method_name,
                    "fold_index": fold_index,
                    "train_doc_ids": " | ".join(sorted(train_docs, key=int)),
                    "test_doc_ids": " | ".join(sorted(test_docs, key=int)),
                    "train_positive_count": int(np.sum(y_train == 1)),
                    "train_negative_count": int(np.sum(y_train == 0)),
                    "binary_threshold": binary_threshold,
                    "tau_filter": tau_filter,
                    "tau_keep": tau_keep,
                    "selected_on_train_only": "true",
                    "threshold_note": threshold_note,
                }
            )

        # Challenge routing with fold-matched models
        challenge_test_rows = [row for row in challenge_rows if row["doc_id"] in test_docs]
        for method_name, method_features, scope in method_definitions:
            model = trained_models[(method_name, fold_index)]
            threshold_row = next(row for row in threshold_rows if row["method_name"] == method_name and row["fold_index"] == fold_index)
            binary_threshold = float(threshold_row["binary_threshold"])
            tau_filter = float(threshold_row["tau_filter"])
            tau_keep = float(threshold_row["tau_keep"])
            if method_name == "no_filter":
                for row in challenge_test_rows:
                    challenge_routing_rows.append(
                        {
                            "candidate_key": row["candidate_key"],
                            "doc_id": row["doc_id"],
                            "challenge_type": row["challenge_type"],
                            "method_name": method_name,
                            "p_correct": 1.0,
                            "p_error": 0.0,
                            "three_way_decision": "keep",
                            "binary_decision": "keep",
                        }
                    )
            else:
                X_challenge = np.asarray([[to_float(row[feature]) for feature in method_features] for row in challenge_test_rows], dtype=float)
                scores = model.predict_proba(X_challenge)[:, 1] if len(challenge_test_rows) else np.asarray([])
                for row, score in zip(challenge_test_rows, scores):
                    binary_decision = "keep" if score >= binary_threshold else "filter"
                    if score < tau_filter:
                        three_way_decision = "filter"
                    elif score < tau_keep:
                        three_way_decision = "review"
                    else:
                        three_way_decision = "keep"
                    challenge_routing_rows.append(
                        {
                            "candidate_key": row["candidate_key"],
                            "doc_id": row["doc_id"],
                            "challenge_type": row["challenge_type"],
                            "method_name": method_name,
                            "p_correct": float(score),
                            "p_error": float(1.0 - score),
                            "three_way_decision": three_way_decision,
                            "binary_decision": binary_decision,
                        }
                    )

    oof_rows.sort(key=lambda row: (row["method_name"], int(row["fold_index"]), row["candidate_key"]))
    write_csv(
        OOF_SCORES_CSV,
        oof_rows,
        [
            "candidate_key",
            "doc_id",
            "fold_index",
            "binary_label",
            "error_code",
            "method_name",
            "p_correct",
            "p_error",
            "binary_threshold",
            "binary_decision",
            "tau_filter",
            "tau_keep",
            "three_way_decision",
        ],
    )
    write_csv(
        THRESHOLDS_CSV,
        threshold_rows,
        [
            "method_name",
            "fold_index",
            "train_doc_ids",
            "test_doc_ids",
            "train_positive_count",
            "train_negative_count",
            "binary_threshold",
            "tau_filter",
            "tau_keep",
            "selected_on_train_only",
            "threshold_note",
        ],
    )

    # Seed filtering metrics and error reduction
    before_pred = np.ones(len(seed_rows), dtype=int)
    y_true_all = np.asarray([int(row["binary_label"]) for row in seed_rows], dtype=int)
    before_scores = np.ones(len(seed_rows), dtype=float)
    before_metrics = compute_metrics(y_true_all, before_pred, before_scores)
    seed_metrics_rows: list[dict[str, Any]] = []
    error_reduction_rows: list[dict[str, Any]] = []

    for method_name, records in seed_method_predictions.items():
        records_sorted = sorted(records, key=lambda row: row["candidate_key"])
        y_true = np.asarray([int(row["binary_label"]) for row in records_sorted], dtype=int)
        p_correct = np.asarray([float(row["p_correct"]) for row in records_sorted], dtype=float)

        for strategy in ["binary", "three_way"]:
            if strategy == "binary":
                pred_keep = np.asarray([1 if row["binary_decision"] == "keep" else 0 for row in records_sorted], dtype=int)
                filtered_count = int(np.sum(pred_keep == 0))
                review_count = 0
                kept_count = int(np.sum(pred_keep == 1))
            else:
                pred_keep = np.asarray([1 if row["three_way_decision"] == "keep" else 0 for row in records_sorted], dtype=int)
                filtered_count = sum(1 for row in records_sorted if row["three_way_decision"] == "filter")
                review_count = sum(1 for row in records_sorted if row["three_way_decision"] == "review")
                kept_count = sum(1 for row in records_sorted if row["three_way_decision"] == "keep")

            after_metrics = compute_metrics(y_true, pred_keep, p_correct)
            fp_before = before_metrics["fp"]
            tp_before = before_metrics["tp"]
            fp_after = after_metrics["fp"]
            tp_after = after_metrics["tp"]
            seed_metrics_rows.append(
                {
                    "method_name": method_name,
                    "threshold_strategy": strategy,
                    "precision_before": before_metrics["precision"],
                    "recall_before": before_metrics["recall"],
                    "f1_before": before_metrics["f1"],
                    "precision_after": after_metrics["precision"],
                    "recall_after": after_metrics["recall"],
                    "f1_after": after_metrics["f1"],
                    "tp_before": before_metrics["tp"],
                    "fp_before": before_metrics["fp"],
                    "tn_before": before_metrics["tn"],
                    "fn_before": before_metrics["fn"],
                    "tp_after": after_metrics["tp"],
                    "fp_after": after_metrics["fp"],
                    "tn_after": after_metrics["tn"],
                    "fn_after": after_metrics["fn"],
                    "fp_reduction_rate": float((fp_before - fp_after) / fp_before) if fp_before else 0.0,
                    "tp_retention_rate": float(tp_after / tp_before) if tp_before else 0.0,
                    "filtered_count": filtered_count,
                    "review_count": review_count,
                    "kept_count": kept_count,
                    "review_rate": float(review_count / len(records_sorted)),
                    "filter_rate": float(filtered_count / len(records_sorted)),
                }
            )

            for error_code in ["E1", "E2", "E3", "E4", "E5"]:
                code_rows = [row for row in records_sorted if row["binary_label"] == "0" and row["error_code"] == error_code]
                support_before = len(code_rows)
                if strategy == "binary":
                    kept_after = sum(1 for row in code_rows if row["binary_decision"] == "keep")
                    filtered_after = sum(1 for row in code_rows if row["binary_decision"] == "filter")
                    review_after = 0
                else:
                    kept_after = sum(1 for row in code_rows if row["three_way_decision"] == "keep")
                    filtered_after = sum(1 for row in code_rows if row["three_way_decision"] == "filter")
                    review_after = sum(1 for row in code_rows if row["three_way_decision"] == "review")
                removed = filtered_after + review_after
                mean_p_error = float(np.mean([float(row["p_error"]) for row in code_rows])) if code_rows else 0.0
                error_reduction_rows.append(
                    {
                        "method_name": method_name,
                        "threshold_strategy": strategy,
                        "error_code": error_code,
                        "support_before": support_before,
                        "kept_after": kept_after,
                        "filtered_after": filtered_after,
                        "review_after": review_after,
                        "error_removed_from_auto_accept": removed,
                        "error_reduction_rate": float(removed / support_before) if support_before else 0.0,
                        "mean_p_error": mean_p_error,
                    }
                )

    write_csv(
        SEED_METRICS_CSV,
        seed_metrics_rows,
        [
            "method_name",
            "threshold_strategy",
            "precision_before",
            "recall_before",
            "f1_before",
            "precision_after",
            "recall_after",
            "f1_after",
            "tp_before",
            "fp_before",
            "tn_before",
            "fn_before",
            "tp_after",
            "fp_after",
            "tn_after",
            "fn_after",
            "fp_reduction_rate",
            "tp_retention_rate",
            "filtered_count",
            "review_count",
            "kept_count",
            "review_rate",
            "filter_rate",
        ],
    )
    write_csv(
        ERROR_REDUCTION_CSV,
        error_reduction_rows,
        [
            "method_name",
            "threshold_strategy",
            "error_code",
            "support_before",
            "kept_after",
            "filtered_after",
            "review_after",
            "error_removed_from_auto_accept",
            "error_reduction_rate",
            "mean_p_error",
        ],
    )

    # Challenge summaries
    write_csv(
        CHALLENGE_ROUTING_CSV,
        challenge_routing_rows,
        ["candidate_key", "doc_id", "challenge_type", "method_name", "p_correct", "p_error", "three_way_decision", "binary_decision"],
    )
    challenge_summary_rows: list[dict[str, Any]] = []
    for method_name in [item[0] for item in method_definitions]:
        for subset in ["benchmark_conflict", "review_invalid_positive", "review_uncertain_positive", "overall"]:
            rows = [row for row in challenge_routing_rows if row["method_name"] == method_name]
            if subset != "overall":
                rows = [row for row in rows if row["challenge_type"] == subset]
            challenge_summary_rows.append(
                {
                    "method_name": method_name,
                    "challenge_subset": subset,
                    "sample_count": len(rows),
                    "binary_keep_count": sum(1 for row in rows if row["binary_decision"] == "keep"),
                    "binary_filter_count": sum(1 for row in rows if row["binary_decision"] == "filter"),
                    "three_way_keep_count": sum(1 for row in rows if row["three_way_decision"] == "keep"),
                    "three_way_review_count": sum(1 for row in rows if row["three_way_decision"] == "review"),
                    "three_way_filter_count": sum(1 for row in rows if row["three_way_decision"] == "filter"),
                    "three_way_review_or_filter_rate": float(
                        sum(1 for row in rows if row["three_way_decision"] in {"review", "filter"}) / len(rows)
                    )
                    if rows
                    else 0.0,
                }
            )
    write_csv(
        CHALLENGE_SUMMARY_CSV,
        challenge_summary_rows,
        [
            "method_name",
            "challenge_subset",
            "sample_count",
            "binary_keep_count",
            "binary_filter_count",
            "three_way_keep_count",
            "three_way_review_count",
            "three_way_filter_count",
            "three_way_review_or_filter_rate",
        ],
    )

    # Open-world routing
    feature_names_for_stability = [
        "feat_cluster_frequency",
        "feat_open_world_name_reuse_count",
        "feat_open_world_parent_reuse_count",
        "feat_open_world_doc_frequency",
        "feat_open_world_cross_model_frequency",
        "feat_evidence_lcs_ratio",
        "feat_definition_has_empty_or_vague_pattern",
        "feat_parent_has_known_level1",
    ]
    normalized_maps: dict[str, list[float]] = {}
    for feature in feature_names_for_stability:
        normalized_maps[feature] = normalize_minmax([to_float(row[feature]) for row in open_world_rows])

    open_world_routing_rows: list[dict[str, Any]] = []
    for idx, row in enumerate(open_world_rows):
        n_cluster = normalized_maps["feat_cluster_frequency"][idx]
        n_name_reuse = normalized_maps["feat_open_world_name_reuse_count"][idx]
        n_parent_reuse = normalized_maps["feat_open_world_parent_reuse_count"][idx]
        n_doc_freq = normalized_maps["feat_open_world_doc_frequency"][idx]
        n_cross_model = normalized_maps["feat_open_world_cross_model_frequency"][idx]
        evidence_lcs = to_float(row["feat_evidence_lcs_ratio"])
        parent_known = to_float(row["feat_parent_has_known_level1"])
        vague = to_float(row["feat_definition_has_empty_or_vague_pattern"])
        stability_score = (
            0.20 * n_cross_model
            + 0.15 * n_doc_freq
            + 0.15 * n_name_reuse
            + 0.15 * n_parent_reuse
            + 0.10 * n_cluster
            + 0.10 * evidence_lcs
            + 0.10 * parent_known
            - 0.05 * vague
        )
        stability_score = min(1.0, max(0.0, stability_score))
        if stability_score >= 0.70:
            route = "potential_valid"
        elif stability_score >= 0.40:
            route = "needs_review"
        else:
            route = "unstable_or_invalid"
        open_world_routing_rows.append(
            {
                "candidate_key": row["candidate_key"],
                "doc_id": row["doc_id"],
                "model_name": row["model_name"],
                "open_world_status": row["open_world_status"],
                "stability_score": stability_score,
                "open_world_route": route,
                "feat_cluster_frequency": to_float(row["feat_cluster_frequency"]),
                "feat_open_world_name_reuse_count": to_float(row["feat_open_world_name_reuse_count"]),
                "feat_open_world_parent_reuse_count": to_float(row["feat_open_world_parent_reuse_count"]),
                "feat_open_world_doc_frequency": to_float(row["feat_open_world_doc_frequency"]),
                "feat_open_world_cross_model_frequency": to_float(row["feat_open_world_cross_model_frequency"]),
                "feat_evidence_lcs_ratio": evidence_lcs,
                "feat_definition_has_empty_or_vague_pattern": vague,
                "feat_parent_has_known_level1": parent_known,
            }
        )
    write_csv(
        OPEN_WORLD_ROUTING_CSV,
        open_world_routing_rows,
        [
            "candidate_key",
            "doc_id",
            "model_name",
            "open_world_status",
            "stability_score",
            "open_world_route",
            "feat_cluster_frequency",
            "feat_open_world_name_reuse_count",
            "feat_open_world_parent_reuse_count",
            "feat_open_world_doc_frequency",
            "feat_open_world_cross_model_frequency",
            "feat_evidence_lcs_ratio",
            "feat_definition_has_empty_or_vague_pattern",
            "feat_parent_has_known_level1",
        ],
    )

    open_world_summary_rows: list[dict[str, Any]] = []
    grouped_open_world: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in open_world_routing_rows:
        grouped_open_world[row["open_world_status"]].append(row)
    for status, rows in sorted(grouped_open_world.items()):
        open_world_summary_rows.append(
            {
                "open_world_status": status,
                "candidate_count": len(rows),
                "potential_valid_count": sum(1 for row in rows if row["open_world_route"] == "potential_valid"),
                "needs_review_count": sum(1 for row in rows if row["open_world_route"] == "needs_review"),
                "unstable_or_invalid_count": sum(1 for row in rows if row["open_world_route"] == "unstable_or_invalid"),
                "mean_stability_score": float(np.mean([row["stability_score"] for row in rows])) if rows else 0.0,
            }
        )
    write_csv(
        OPEN_WORLD_SUMMARY_CSV,
        open_world_summary_rows,
        ["open_world_status", "candidate_count", "potential_valid_count", "needs_review_count", "unstable_or_invalid_count", "mean_stability_score"],
    )

    # Method comparison summary
    method_comparison_rows: list[dict[str, Any]] = []
    for row in seed_metrics_rows:
        method_comparison_rows.append(dict(row))
    write_csv(
        METHOD_COMPARISON_CSV,
        method_comparison_rows,
        [
            "method_name",
            "threshold_strategy",
            "precision_before",
            "recall_before",
            "f1_before",
            "precision_after",
            "recall_after",
            "f1_after",
            "tp_before",
            "fp_before",
            "tn_before",
            "fn_before",
            "tp_after",
            "fp_after",
            "tn_after",
            "fn_after",
            "fp_reduction_rate",
            "tp_retention_rate",
            "filtered_count",
            "review_count",
            "kept_count",
            "review_rate",
            "filter_rate",
        ],
    )

    # Audit + manifest
    best_row = max(seed_metrics_rows, key=lambda row: (float(row["f1_after"]), float(row["precision_after"])))
    full_binary = next(row for row in seed_metrics_rows if row["method_name"] == "full_diagnostic_v2_logreg" and row["threshold_strategy"] == "binary")
    full_3way = next(row for row in seed_metrics_rows if row["method_name"] == "full_diagnostic_v2_logreg" and row["threshold_strategy"] == "three_way")
    full_wo_evidence = next(row for row in seed_metrics_rows if row["method_name"] == "full_without_evidence_v2_logreg" and row["threshold_strategy"] == "binary")
    consensus_only = next(row for row in seed_metrics_rows if row["method_name"] == "consensus_v2_only_logreg" and row["threshold_strategy"] == "binary")
    error_best = max(
        [row for row in error_reduction_rows if row["threshold_strategy"] == "three_way"],
        key=lambda row: float(row["error_reduction_rate"]),
    )
    challenge_overall_full = next(row for row in challenge_summary_rows if row["method_name"] == "full_diagnostic_v2_logreg" and row["challenge_subset"] == "overall")
    route_counts = Counter(row["open_world_route"] for row in open_world_routing_rows)
    audit_lines = [
        "# RQ3 Filtering Audit",
        "",
        f"- Generated at: {RUN_TS}",
        f"- Best RQ3 method/strategy by F1_after: {best_row['method_name']} / {best_row['threshold_strategy']}",
        f"- Full Diagnostic-v2 binary before P/R/F1: {float(full_binary['precision_before']):.6f}/{float(full_binary['recall_before']):.6f}/{float(full_binary['f1_before']):.6f}",
        f"- Full Diagnostic-v2 binary after P/R/F1: {float(full_binary['precision_after']):.6f}/{float(full_binary['recall_after']):.6f}/{float(full_binary['f1_after']):.6f}",
        f"- Full Diagnostic-v2 three-way after P/R/F1: {float(full_3way['precision_after']):.6f}/{float(full_3way['recall_after']):.6f}/{float(full_3way['f1_after']):.6f}",
        f"- Full without Evidence-v2 binary after P/R/F1: {float(full_wo_evidence['precision_after']):.6f}/{float(full_wo_evidence['recall_after']):.6f}/{float(full_wo_evidence['f1_after']):.6f}",
        f"- Consensus-v2-only binary after P/R/F1: {float(consensus_only['precision_after']):.6f}/{float(consensus_only['recall_after']):.6f}/{float(consensus_only['f1_after']):.6f}",
        f"- Full Diagnostic-v2 FP reduction rate / TP retention rate (binary): {float(full_binary['fp_reduction_rate']):.6f} / {float(full_binary['tp_retention_rate']):.6f}",
        f"- Full Diagnostic-v2 FP reduction rate / TP retention rate (three-way): {float(full_3way['fp_reduction_rate']):.6f} / {float(full_3way['tp_retention_rate']):.6f}",
        f"- Error type with largest reduction under three-way: {error_best['error_code']} ({float(error_best['error_reduction_rate']):.6f})",
        f"- Challenge overall review/filter rate for Full Diagnostic-v2: {float(challenge_overall_full['three_way_review_or_filter_rate']):.6f}",
        f"- Open-world route counts: potential_valid={route_counts['potential_valid']}, needs_review={route_counts['needs_review']}, unstable_or_invalid={route_counts['unstable_or_invalid']}",
        "- Consensus-v2 features used in training, challenge routing, and open-world supplementation are all derived from `unlabeled_candidate_universe.csv`.",
        "- Recommended as RQ3 main result: yes, because the setup is OOF by document, thresholds are chosen on train folds only, and the pipeline is consistent with the final Consensus-v2 RQ2 model line.",
    ]
    AUDIT_MD.write_text("\n".join(audit_lines) + "\n", encoding="utf-8")

    manifest = {
        "created_at_utc": RUN_TS,
        "run_script": rel(Path(__file__)),
        "seed": SEED,
        "selected_features": selected_features,
        "input_files": {
            rel(SEED_FEATURES_CSV): sha256_file(SEED_FEATURES_CSV),
            rel(FEATURE_COLUMNS_JSON): sha256_file(FEATURE_COLUMNS_JSON),
            rel(FOLDS_JSON): sha256_file(FOLDS_JSON),
            rel(UNIVERSE_CSV): sha256_file(UNIVERSE_CSV),
            rel(CHALLENGE_FEATURES_CSV): sha256_file(CHALLENGE_FEATURES_CSV),
            rel(OPEN_WORLD_FEATURES_CSV): sha256_file(OPEN_WORLD_FEATURES_CSV),
            rel(OPEN_WORLD_POOL_CSV): sha256_file(OPEN_WORLD_POOL_CSV),
            rel(RQ2_SUMMARY_CSV): sha256_file(RQ2_SUMMARY_CSV),
            rel(RQ2_ABLATION_CSV): sha256_file(RQ2_ABLATION_CSV),
        },
        "output_files": {},
    }
    for output_path in [
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
    ]:
        manifest["output_files"][rel(output_path)] = sha256_file(output_path)
    MANIFEST_JSON.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    VALIDATION_JSON.write_text(
        json.dumps({"status": "pending_validation", "generated_at_utc": RUN_TS}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"seed_oof_rows={len(oof_rows)}")
    print(f"challenge_rows={len(challenge_routing_rows)}")
    print(f"open_world_rows={len(open_world_routing_rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
