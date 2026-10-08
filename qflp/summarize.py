"""Count attack outcomes per setting from sweep JSONL files.

    uv run python -m qflp.summarize results/rq1.jsonl

Each seed is classified by its best restart, the one with the lowest matching loss, which is what
an attacker without the ground truth would keep.
"""

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

SETTING = ("n_qubits", "reps", "layers", "shots", "batch", "restarts")
OUTCOMES = ("recovered", "ambiguous", "stuck")
# Under shot noise the exact input is out of reach; a reconstruction this close still leaks it.
CLOSE_RAD = 0.2


def load(paths: list[Path]) -> list[dict]:
    rows = []
    for p in paths:
        with p.open() as fh:
            rows.extend(json.loads(line) for line in fh)
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
        " ambiguous | stuck | any restart recovered | median error (rad) |"
        f" within {CLOSE_RAD} rad |"
    )
    lines = [head, "|" + "---|" * 12]
    for s in summary:
        shots = s["shots"] or "exact"
        lines.append(
            f"| {s['n_qubits']} | {s['reps']} x {s['layers']} | {s['n_effective']} | {shots} | "
            f"{s['batch']} | {s['seeds']} | {s['recovered']} | {s['ambiguous']} | {s['stuck']} | "
            f"{s['any_restart_recovered']} | {s['median_error']:.3f} | {s['within_close']} |"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("paths", type=Path, nargs="+")
    args = ap.parse_args(argv)
    print(markdown(table(load(args.paths))))  # noqa: T201


if __name__ == "__main__":
    main()
