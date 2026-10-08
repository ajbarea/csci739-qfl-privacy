"""Figures and tables for the report, drawn from the sweep JSONL files in results/.

uv run python -m qflp.figures

Writes report/figures/*.pdf and report/tables/*.tex. The report inputs the tables, so no number in
them is typed by hand.
"""

import argparse
import math
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from qflp.summarize import CLOSE_RAD, load, table

# Okabe-Ito, checked for colour-vision deficiency; markers repeat the identity for print.
STYLE = [("#0072B2", "o"), ("#D55E00", "s"), ("#009E73", "^")]


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for k successes in n trials."""
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(0.0, centre - half), min(1.0, centre + half)


def privacy_map(rows: list[dict], out: Path) -> None:
    """Recovered fraction against encoding reps, one line per fixed reps x layers product."""
    exact = [r for r in rows if r["shots"] == 0 and r["batch"] == 1]
    qubits = sorted({r["n_qubits"] for r in exact})
    fig, axes = plt.subplots(1, len(qubits), figsize=(3.5 * len(qubits), 2.9), sharey=True)
    axes = [axes] if len(qubits) == 1 else list(axes)
    for ax, n in zip(axes, qubits, strict=True):
        groups: dict[int, dict[int, list[bool]]] = defaultdict(lambda: defaultdict(list))
        for r in exact:
            if r["n_qubits"] == n and r["reps"] * r["layers"] > 1:
                groups[r["reps"] * r["layers"]][r["reps"]].append(r["best_outcome"] == "recovered")
        for i, ((product, by_reps), (colour, marker)) in enumerate(
            zip(sorted(groups.items()), STYLE, strict=False)
        ):
            reps = sorted(by_reps)
            k = [sum(by_reps[x]) for x in reps]
            m = [len(by_reps[x]) for x in reps]
            frac = [a / b for a, b in zip(k, m, strict=True)]
            lo, hi = zip(*(wilson(a, b) for a, b in zip(k, m, strict=True)), strict=True)
            yerr = [
                [f - lo_ for f, lo_ in zip(frac, lo, strict=True)],
                [hi_ - f for f, hi_ in zip(frac, hi, strict=True)],
            ]
            params = 2 * n * product - n
            dodge = 2 ** (0.05 * (i - 1))  # separate overlapping error bars
            ax.errorbar(
                [x * dodge for x in reps],
                frac,
                yerr=yerr,
                color=colour,
                marker=marker,
                ms=6,
                lw=2,
                capsize=3,
                label=f"{params} effective params",
            )
        ax.set_xscale("log", base=2)
        ax.set_xticks([1, 2, 4, 8], labels=["1", "2", "4", "8"])
        ax.set_xlabel("encoding repetitions $r$ (layers $= $ product$/r$)")
        ax.set_title(f"{n} qubits", fontsize=10)
        ax.set_ylim(-0.03, 1.03)
        ax.grid(alpha=0.25)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.legend(frameon=False, fontsize=8, loc="lower left")
    axes[0].set_ylabel("seeds recovered (fraction)")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)


def shot_noise(rows: list[dict], out: Path) -> None:
    """Median reconstruction error against shots per gradient entry, one line per circuit."""
    qubits = sorted({r["n_qubits"] for r in rows})
    fig, axes = plt.subplots(1, len(qubits), figsize=(3.5 * len(qubits), 2.9), sharey=True)
    axes = [axes] if len(qubits) == 1 else list(axes)
    for ax, n in zip(axes, qubits, strict=True):
        grids = sorted(
            {(r["reps"], r["layers"]) for r in rows if r["n_qubits"] == n}, key=lambda g: g[0]
        )
        for (reps, layers), (colour, marker) in zip(grids, STYLE, strict=False):
            sel = [
                s
                for s in table(rows)
                if (s["n_qubits"], s["reps"], s["layers"]) == (n, reps, layers)
            ]
            sel.sort(key=lambda s: s["shots"])
            ax.plot(
                [s["shots"] for s in sel],
                [s["median_error"] for s in sel],
                color=colour,
                marker=marker,
                ms=6,
                lw=2,
                label=f"{reps} x {layers}",
            )
        ax.axhline(CLOSE_RAD, color="#6b6b6b", lw=1, ls="--")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("shots per expectation value")
        ax.set_title(f"{n} qubits", fontsize=10)
        ax.grid(alpha=0.25)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.legend(frameon=False, fontsize=8, title="reps x layers", title_fontsize=8)
    axes[0].set_ylabel("median error (rad)")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)


def outcome_table(rows: list[dict], out: Path) -> None:
    """LaTeX rows: qubits, reps x layers, effective params, recovered / ambiguous / stuck."""
    lines = [
        f"{s['n_qubits']} & ${s['reps']}\\times{s['layers']}$ & {s['n_effective']} & "
        f"{s['recovered']} & {s['ambiguous']} & {s['stuck']} \\\\"
        for s in table(rows)
    ]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")


def shots_table(rows: list[dict], out: Path) -> None:
    """LaTeX rows: qubits, reps x layers, then median error and within-CLOSE_RAD count per shots."""
    summary = table(rows)
    lines = []
    for n, reps, layers in sorted({(s["n_qubits"], s["reps"], s["layers"]) for s in summary}):
        sel = sorted(
            (s for s in summary if (s["n_qubits"], s["reps"], s["layers"]) == (n, reps, layers)),
            key=lambda s: s["shots"],
        )
        cells = " & ".join(f"{s['median_error']:.3f} ({s['within_close']})" for s in sel)
        lines.append(f"{n} & ${reps}\\times{layers}$ & {cells} \\\\")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--results", type=Path, default=Path("results"))
    ap.add_argument("--report", type=Path, default=Path("report"))
    args = ap.parse_args(argv)
    rq1 = load([args.results / "rq1.jsonl"])
    rq2 = load([args.results / "rq2.jsonl"])
    privacy_map(rq1, args.report / "figures" / "privacy-map.pdf")
    outcome_table(rq1, args.report / "tables" / "rq1.tex")
    shot_noise(rq2, args.report / "figures" / "shot-noise.pdf")
    shots_table(rq2, args.report / "tables" / "rq2.tex")
    budget = args.results / "budget.jsonl"
    if budget.exists():
        outcome_table(load([budget]), args.report / "tables" / "budget.tex")


if __name__ == "__main__":
    main()
