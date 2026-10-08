"""Figures for the report, drawn from the sweep JSONL files.

uv run python -m qflp.figures results/rq1.jsonl --out report/figures/privacy-map.pdf
"""

import argparse
import math
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from qflp.summarize import load

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


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("paths", type=Path, nargs="+")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    privacy_map(load(args.paths), args.out)


if __name__ == "__main__":
    main()
