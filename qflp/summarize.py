"""Count attack outcomes per setting from sweep JSONL files.

    uv run python -m qflp.summarize results/rq1.jsonl

Each seed is classified by its best restart, the one with the lowest matching loss, which is what
an attacker without the ground truth would keep. Outcomes are recomputed here from each restart's
raw error and loss with `qflp.attack.classify`, so a threshold change never needs a rerun.
"""

import argparse
import json
import os
import statistics
from collections import defaultdict
from pathlib import Path

from qflp.attack import CLOSE_RAD, OUTCOMES, classify

SETTING = ("n_qubits", "reps", "layers", "shots", "batch", "restarts")
KEY = (*SETTING, "seed")  # one row per run


def read_rows(path: Path, repair: bool = False) -> list[dict]:
    """Every row of a sweep file.

    Only a partial last line (no trailing newline, as a killed sweep leaves) is tolerated: it is
    skipped, and with `repair` removed from the file by an atomic rewrite. Any other line that is
    not JSON raises, so a non-sweep file is never silently emptied.
    """
    lines = path.read_text().splitlines(keepends=True)
    rows = []
    for i, line in enumerate(lines):
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            if i == len(lines) - 1 and not line.endswith("\n"):
                if repair:
                    tmp = path.with_name(path.name + ".tmp")
                    tmp.write_text("".join(lines[:-1]))
                    os.replace(tmp, path)
                break
            raise ValueError(f"{path}:{i + 1} is not a sweep row") from None
    return rows


def outcomes(row: dict) -> list[str]:
    """Outcome of every restart in a sweep row, by the current `classify` thresholds."""
    return [
        classify(d["error"], d["match"], row["true_match"], row["grad_norm2"])
        for d in row["restarts_detail"]
    ]


def best_outcome(row: dict, budget: int | None = None) -> str:
    """Outcome of the lowest-loss restart among the first `budget` (default: all)."""
    detail = row["restarts_detail"][:budget]
    best = min(range(len(detail)), key=lambda i: detail[i]["match"])
    return outcomes(row)[best]


def load(paths: list[Path]) -> list[dict]:
    rows = [r for p in paths for r in read_rows(p)]
    keys = [tuple(r[k] for k in KEY) for r in rows]
    if len(set(keys)) != len(keys):
        raise ValueError("duplicate runs across the given files; counts would double")
    for r in rows:
        r["best_outcome"] = best_outcome(r)
        r["any_recovered"] = "recovered" in outcomes(r)
    return rows


def table(rows: list[dict]) -> list[dict]:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        groups[tuple(r[k] for k in SETTING)].append(r)
    out = []
    for key, rs in sorted(groups.items()):
        counts = {o: sum(r["best_outcome"] == o for r in rs) for o in OUTCOMES}
        out.append(
            {
                **dict(zip(SETTING, key, strict=True)),
                "n_effective": rs[0]["n_effective"],
                "seeds": len(rs),
                **counts,
                "any_restart_recovered": sum(r["any_recovered"] for r in rs),
                "median_error": statistics.median(r["best_error"] for r in rs),
                "within_close": sum(r["best_error"] <= CLOSE_RAD for r in rs),
            }
        )
    return out


def markdown(summary: list[dict]) -> str:
    head = (
        "| qubits | reps x layers | effective params | shots | batch | seeds | recovered |"
        f" close (<= {CLOSE_RAD} rad) | ambiguous | stuck | any restart recovered |"
        " median error (rad) |"
    )
    lines = [head, "|" + "---|" * 12]
    for s in summary:
        shots = s["shots"] or "exact"
        lines.append(
            f"| {s['n_qubits']} | {s['reps']} x {s['layers']} | {s['n_effective']} | {shots} | "
            f"{s['batch']} | {s['seeds']} | {s['recovered']} | {s['close']} | {s['ambiguous']} | "
            f"{s['stuck']} | {s['any_restart_recovered']} | {s['median_error']:.3f} |"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("paths", type=Path, nargs="+")
    args = ap.parse_args(argv)
    print(markdown(table(load(args.paths))))  # noqa: T201


if __name__ == "__main__":
    main()
