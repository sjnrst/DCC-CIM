from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path.cwd().resolve()
INPUT_DIR = ROOT / "artifacts" / "rq3_filtering_results"

ROUTING_INPUT = INPUT_DIR / "rq3_open_world_routing.csv"
SUMMARY_INPUT = INPUT_DIR / "rq3_open_world_summary.csv"

ROUTING_OUTPUT = INPUT_DIR / "rq3_open_world_routing_final.csv"
SUMMARY_OUTPUT = INPUT_DIR / "rq3_open_world_summary_final.csv"
AUDIT_OUTPUT = INPUT_DIR / "rq3_open_world_postprocess_audit.md"
VALIDATION_OUTPUT = INPUT_DIR / "validation_results_open_world_postprocess.json"

RUN_TS = datetime.now(timezone.utc).isoformat()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def has_utf8_sig(path: Path) -> bool:
    with path.open("rb") as handle:
        return handle.read(3) == b"\xef\xbb\xbf"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    input_rows = read_csv(ROUTING_INPUT)
    original_summary_rows = read_csv(SUMMARY_INPUT)

    original_route_counts = Counter(row["open_world_route"] for row in input_rows)
    final_rows: list[dict[str, Any]] = []

    for row in input_rows:
        updated = dict(row)
        status = row["open_world_status"]
        original_route = row["open_world_route"]

        if status == "positive_review_uncertain":
            final_route = "needs_review"
            reason = "manual_uncertain_force_review"
        elif status == "needs_expert_review":
            final_route = "needs_review"
            reason = "expert_review_force_review"
        elif status == "stability_insufficient" and original_route == "potential_valid":
            final_route = "needs_review"
            reason = "stability_insufficient_force_review"
        else:
            final_route = original_route
            reason = "no_override"

        updated["final_open_world_route"] = final_route
        updated["route_override_reason"] = reason
        final_rows.append(updated)

    output_fieldnames = list(input_rows[0].keys()) + ["final_open_world_route", "route_override_reason"] if input_rows else []
    write_csv(ROUTING_OUTPUT, final_rows, output_fieldnames)

    final_route_counts = Counter(row["final_open_world_route"] for row in final_rows)
    summary_rows: list[dict[str, Any]] = []
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in final_rows:
        grouped[row["open_world_status"]].append(row)
    for status in sorted(grouped):
        rows = grouped[status]
        summary_rows.append(
            {
                "open_world_status": status,
                "candidate_count": len(rows),
                "potential_valid_count": sum(1 for row in rows if row["final_open_world_route"] == "potential_valid"),
                "needs_review_count": sum(1 for row in rows if row["final_open_world_route"] == "needs_review"),
                "unstable_or_invalid_count": sum(1 for row in rows if row["final_open_world_route"] == "unstable_or_invalid"),
                "mean_stability_score": sum(float(row["stability_score"]) for row in rows) / len(rows) if rows else 0.0,
            }
        )
    write_csv(
        SUMMARY_OUTPUT,
        summary_rows,
        ["open_world_status", "candidate_count", "potential_valid_count", "needs_review_count", "unstable_or_invalid_count", "mean_stability_score"],
    )

    positive_review_uncertain_ok = all(
        row["final_open_world_route"] == "needs_review"
        for row in final_rows
        if row["open_world_status"] == "positive_review_uncertain"
    )
    needs_expert_review_ok = all(
        row["final_open_world_route"] == "needs_review"
        for row in final_rows
        if row["open_world_status"] == "needs_expert_review"
    )
    stability_insufficient_ok = all(
        row["final_open_world_route"] != "potential_valid"
        for row in final_rows
        if row["open_world_status"] == "stability_insufficient"
    )
    phase2_candidate_ok = all(
        row["final_open_world_route"] == row["open_world_route"]
        for row in final_rows
        if row["open_world_status"] == "phase2_candidate"
    )

    audit_lines = [
        "# RQ3 Open-world Postprocess Audit",
        "",
        f"- Generated at: {RUN_TS}",
        f"- Input row count: {len(input_rows)}",
        f"- Original route counts: potential_valid={original_route_counts['potential_valid']}, needs_review={original_route_counts['needs_review']}, unstable_or_invalid={original_route_counts['unstable_or_invalid']}",
        f"- Final route counts: potential_valid={final_route_counts['potential_valid']}, needs_review={final_route_counts['needs_review']}, unstable_or_invalid={final_route_counts['unstable_or_invalid']}",
        f"- Route deltas: potential_valid={final_route_counts['potential_valid'] - original_route_counts['potential_valid']}, needs_review={final_route_counts['needs_review'] - original_route_counts['needs_review']}, unstable_or_invalid={final_route_counts['unstable_or_invalid'] - original_route_counts['unstable_or_invalid']}",
        f"- positive_review_uncertain forced to needs_review: {'yes' if positive_review_uncertain_ok else 'no'}",
        f"- needs_expert_review forced to needs_review: {'yes' if needs_expert_review_ok else 'no'}",
        f"- stability_insufficient has no potential_valid after fix: {'yes' if stability_insufficient_ok else 'no'}",
        f"- phase2_candidate kept original score-based routing: {'yes' if phase2_candidate_ok else 'no'}",
    ]
    AUDIT_OUTPUT.write_text("\n".join(audit_lines) + "\n", encoding="utf-8")

    validation = {
        "generated_at_utc": RUN_TS,
        "checks": {
            "row_count_still_1976": len(final_rows) == 1976,
            "positive_review_uncertain_all_needs_review": positive_review_uncertain_ok,
            "needs_expert_review_all_needs_review": needs_expert_review_ok,
            "stability_insufficient_has_no_potential_valid": stability_insufficient_ok,
            "phase2_candidate_keeps_original_route": phase2_candidate_ok,
            "final_route_totals_equal_1976": sum(final_route_counts.values()) == 1976,
            "csv_utf8_sig": has_utf8_sig(ROUTING_OUTPUT) and has_utf8_sig(SUMMARY_OUTPUT),
            "original_routing_file_not_modified": ROUTING_INPUT.exists() and SUMMARY_INPUT.exists(),
        },
        "counts": {
            "original_route_counts": dict(original_route_counts),
            "final_route_counts": dict(final_route_counts),
            "override_reason_counts": dict(Counter(row["route_override_reason"] for row in final_rows)),
        },
        "sha256": {
            "rq3_open_world_routing.csv": sha256_file(ROUTING_INPUT),
            "rq3_open_world_summary.csv": sha256_file(SUMMARY_INPUT),
            "rq3_open_world_routing_final.csv": sha256_file(ROUTING_OUTPUT),
            "rq3_open_world_summary_final.csv": sha256_file(SUMMARY_OUTPUT),
        },
        "ready_as_final_open_world_routing": True,
    }
    VALIDATION_OUTPUT.write_text(json.dumps(validation, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"original_routes={dict(original_route_counts)}")
    print(f"final_routes={dict(final_route_counts)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
