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

from qflp.summarize import CLOSE_RAD, best_outcome, load, outcomes, table

# Okabe-Ito, checked for colour-vision deficiency; markers repeat the identity for print.
STYLE = [("#0072B2", "o"), ("#D55E00", "s"), ("#009E73", "^")]


def mcnemar_exact(only_a: int, only_b: int) -> float:
    """Two-sided exact McNemar p-value from the two discordant counts of a paired comparison."""
    n = only_a + only_b
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(only_a, only_b) + 1))
    return min(1.0, 2 * tail / 2**n)


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for k successes in n trials."""
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(0.0, centre - half), min(1.0, centre + half)


def _styles(n_series: int) -> list[tuple[str, str]]:
    if n_series > len(STYLE):
        raise ValueError(f"{n_series} series but only {len(STYLE)} validated styles")
    return STYLE[:n_series]


def _tidy(ax) -> None:
    ax.grid(alpha=0.25)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def _save(fig, out: Path) -> None:
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, metadata={"CreationDate": None})  # byte-stable across reruns
    plt.close(fig)


def _recovered_with_interval(flags: list[bool]) -> tuple[float, float, float]:
    """Fraction recovered and the distances down and up to its Wilson interval."""
    k, m = sum(flags), len(flags)
    lo, hi = wilson(k, m)
    return k / m, k / m - lo, hi - k / m


def privacy_map(rows: list[dict], out: Path, restarts: int = 10) -> None:
    """Recovered fraction against encoding reps, one line per fixed reps x layers product."""
    exact = [r for r in rows if r["shots"] == 0 and r["batch"] == 1 and r["restarts"] == restarts]
    qubits = sorted({r["n_qubits"] for r in exact})
    fig, axes = plt.subplots(1, len(qubits), figsize=(3.5 * len(qubits), 2.9), sharey=True)
    axes = [axes] if len(qubits) == 1 else list(axes)
    for ax, n in zip(axes, qubits, strict=True):
        groups: dict[int, dict[int, list[bool]]] = defaultdict(lambda: defaultdict(list))
        params: dict[int, int] = {}
        for r in exact:
            if r["n_qubits"] == n and r["reps"] * r["layers"] > 1:
                product = r["reps"] * r["layers"]
                groups[product][r["reps"]].append(r["best_outcome"] == "recovered")
                params[product] = r["n_effective"]
        series = sorted(groups.items())
        for i, ((product, by_reps), (colour, marker)) in enumerate(
            zip(series, _styles(len(series)), strict=True)
        ):
            reps = sorted(by_reps)
            stats = [_recovered_with_interval(by_reps[x]) for x in reps]
            frac = [f for f, _, _ in stats]
            yerr = [[d for _, d, _ in stats], [u for _, _, u in stats]]
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
                label=f"{params[product]} effective params",
            )
        ax.set_xscale("log", base=2)
        ax.set_xticks([1, 2, 4, 8], labels=["1", "2", "4", "8"])
        ax.set_xlabel("encoding repetitions $r$ (layers $= $ product$/r$)")
        ax.set_title(f"{n} qubits", fontsize=10)
        ax.set_ylim(-0.03, 1.03)
        _tidy(ax)
        ax.legend(frameon=False, fontsize=8, loc="lower left")
    axes[0].set_ylabel("seeds recovered (fraction)")
    _save(fig, out)


def shot_noise(rows: list[dict], out: Path) -> None:
    """Median reconstruction error against shots per gradient entry, one line per circuit."""
    rows = [r for r in rows if r["batch"] == 1]
    summary = table(rows)
    qubits = sorted({r["n_qubits"] for r in rows})
    fig, axes = plt.subplots(1, len(qubits), figsize=(3.5 * len(qubits), 2.9), sharey=True)
    axes = [axes] if len(qubits) == 1 else list(axes)
    for ax, n in zip(axes, qubits, strict=True):
        grids = sorted(
            {(r["reps"], r["layers"]) for r in rows if r["n_qubits"] == n}, key=lambda g: g[0]
        )
        for (reps, layers), (colour, marker) in zip(grids, _styles(len(grids)), strict=True):
            sel = [
                s for s in summary if (s["n_qubits"], s["reps"], s["layers"]) == (n, reps, layers)
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
        _tidy(ax)
        ax.legend(frameon=False, fontsize=8, title="reps x layers", title_fontsize=8)
    axes[0].set_ylabel("median error (rad)")
    _save(fig, out)


def outcome_table(rows: list[dict], out: Path) -> None:
    """LaTeX rows: qubits, reps x layers, effective params, then the four outcome counts."""
    lines = [
        f"{s['n_qubits']} & ${s['reps']}\\times{s['layers']}$ & {s['n_effective']} & "
        f"{s['recovered']} & {s['close']} & {s['ambiguous']} & {s['stuck']} \\\\"
        for s in table(rows)
    ]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")


# Columns of the report's RQ2 table header; the generated rows must fill exactly these.
SHOTS = (10**2, 10**3, 10**4, 10**5, 10**6)


def shots_table(rows: list[dict], out: Path) -> None:
    """LaTeX rows: qubits, reps x layers, then median error and within-CLOSE_RAD count per shots."""
    summary = table([r for r in rows if r["batch"] == 1])
    lines = []
    for n, reps, layers in sorted({(s["n_qubits"], s["reps"], s["layers"]) for s in summary}):
        sel = sorted(
            (s for s in summary if (s["n_qubits"], s["reps"], s["layers"]) == (n, reps, layers)),
            key=lambda s: s["shots"],
        )
        if tuple(s["shots"] for s in sel) != SHOTS:
            raise ValueError(f"{n}q {reps}x{layers}: shots {[s['shots'] for s in sel]} != {SHOTS}")
        cells = " & ".join(f"{s['median_error']:.3f} ({s['within_close']})" for s in sel)
        lines.append(f"{n} & ${reps}\\times{layers}$ & {cells} \\\\")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")


def budget_table(rows: list[dict], out: Path, budgets: tuple[int, ...] = (10, 50)) -> None:
    """LaTeX rows: qubits, reps x layers, effective params, seeds recovered within each restart
    budget, and the fraction of all restarts that recover the input."""
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        if r["shots"] == 0 and r["batch"] == 1:
            groups[(r["n_qubits"], r["reps"], r["layers"])].append(r)
    lines = []
    for (n, reps, layers), rs in sorted(groups.items()):
        if min(r["restarts"] for r in rs) < max(budgets):
            raise ValueError(f"{n}q {reps}x{layers}: fewer restarts than the {max(budgets)} budget")
        within = [sum(best_outcome(r, b) == "recovered" for r in rs) for b in budgets]
        per = [o == "recovered" for r in rs for o in outcomes(r)]
        cells = " & ".join(f"{w}/{len(rs)}" for w in within)
        lines.append(
            f"{n} & ${reps}\\times{layers}$ & {rs[0]['n_effective']} & {cells} & "
            f"{100 * sum(per) / len(per):.1f}\\% \\\\"
        )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")


def _exact_fixed(rows: list[dict], restarts: int = 10) -> list[dict]:
    return [
        r for r in rows if r["shots"] == 0 and r["label"] == "fixed" and r["restarts"] == restarts
    ]


def batch_map(rows: list[dict], out: Path) -> None:
    """Recovered fraction against unknowns per effective parameter (N·B / P_eff), one line per
    circuit, so circuits of different size share the point where the inputs outnumber the
    equations."""
    rows = _exact_fixed(rows)
    qubits = sorted({r["n_qubits"] for r in rows})
    fig, axes = plt.subplots(1, len(qubits), figsize=(3.5 * len(qubits), 2.9), sharey=True)
    axes = [axes] if len(qubits) == 1 else list(axes)
    for ax, n in zip(axes, qubits, strict=True):
        grids = sorted({(r["reps"], r["layers"]) for r in rows if r["n_qubits"] == n})
        for i, ((reps, layers), (colour, marker)) in enumerate(
            zip(grids, _styles(len(grids)), strict=True)
        ):
            sel = [r for r in rows if (r["n_qubits"], r["reps"], r["layers"]) == (n, reps, layers)]
            batches = sorted({r["batch"] for r in sel})
            n_eff = sel[0]["n_effective"]
            stats = [
                _recovered_with_interval(
                    [r["best_outcome"] == "recovered" for r in sel if r["batch"] == b]
                )
                for b in batches
            ]
            dodge = 2 ** (0.05 * (i - 0.5))  # separate overlapping error bars
            ax.errorbar(
                [n * b / n_eff * dodge for b in batches],
                [f for f, _, _ in stats],
                yerr=[[d for _, d, _ in stats], [u for _, _, u in stats]],
                color=colour,
                marker=marker,
                ms=6,
                lw=2,
                capsize=3,
                label=f"{reps} x {layers} ({n_eff} effective)",
            )
        ax.axvline(1, color="#6b6b6b", lw=1, ls="--")
        ax.set_xscale("log", base=2)
        ax.set_xticks([0.25, 0.5, 1, 2], labels=["1/4", "1/2", "1", "2"])
        ax.set_xlabel("unknown inputs per effective parameter")
        ax.set_title(f"{n} qubits", fontsize=10)
        ax.set_ylim(-0.03, 1.03)
        _tidy(ax)
        ax.legend(frameon=False, fontsize=8, loc="lower left")
    axes[0].set_ylabel("seeds recovered (fraction)")
    _save(fig, out)


def batch_table(rows: list[dict], out: Path) -> dict[str, str]:
    """LaTeX rows: qubits, reps x layers, effective params, batch, unknowns, four outcome counts,
    and the median error of the worst-matched input.

    Returns report macros: the largest unknowns-per-parameter ratio with any recovery, the
    smallest with none, the fewest seeds recovered below a ratio of 1, and the range of median
    errors among settings past it."""
    summary = table(_exact_fixed(rows))
    ratio = {id(s): s["n_qubits"] * s["batch"] / s["n_effective"] for s in summary}
    leaked = [s for s in summary if s["recovered"] > 0]
    held = [s for s in summary if s["recovered"] == 0]
    if max(ratio[id(s)] for s in leaked) >= min(ratio[id(s)] for s in held):
        raise ValueError("recovery does not separate on unknowns per effective parameter")
    errors = [s["median_error"] for s in summary if ratio[id(s)] > 1]
    lines = [
        f"{s['n_qubits']} & ${s['reps']}\\times{s['layers']}$ & {s['n_effective']} & "
        f"{s['batch']} & {s['n_qubits'] * s['batch']} & "
        f"{s['recovered']} & {s['close']} & {s['ambiguous']} & {s['stuck']} & "
        f"{s['median_error']:.2f} \\\\"
        for s in summary
    ]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")
    return {
        "BatchLeakMaxRatio": f"{max(ratio[id(s)] for s in leaked):.2f}",
        "BatchHeldMinRatio": f"{min(ratio[id(s)] for s in held):.2f}",
        "BatchFewestRecoveredBelow": str(min(s["recovered"] for s in summary if ratio[id(s)] < 1)),
        "BatchAmbigErrLow": f"{min(errors):.1f}",
        "BatchAmbigErrHigh": f"{max(errors):.1f}",
    }


def label_table(rows: list[dict], out: Path) -> dict[str, str]:
    """LaTeX rows: qubits, reps x layers, batch, recovered/close/ambiguous/stuck with the label
    known and with it unknown on the same clients, and the exact McNemar p-value for recovery.

    Returns report macros for the same test pooled over every setting (the primary test) and, as
    description, per batch size, with the largest single setting's share of the known-only seeds.
    Clients are independent across settings, so their discordant pairs add."""
    rows = [r for r in rows if r["shots"] == 0]
    setting = ("n_qubits", "reps", "layers", "batch", "restarts")
    summary = {(*(s[k] for k in setting), s["label"]): s for s in table(rows)}
    recovered = {
        (*(r[k] for k in setting), r["label"], r["seed"]): r["best_outcome"] == "recovered"
        for r in rows
    }
    lines = []
    pooled: dict[int, list[int]] = defaultdict(lambda: [0, 0])
    top: dict[int, tuple[int, str]] = defaultdict(lambda: (0, ""))
    for key in sorted({k[:-1] for k in summary}):
        n, reps, layers, b, _ = key
        cells = []
        for label in ("known", "unknown"):
            s = summary[(*key, label)]
            cells.append(f"{s['recovered']}/{s['close']}/{s['ambiguous']}/{s['stuck']}")
        seeds = sorted({k[-1] for k in recovered if k[:5] == key})
        pairs = [(recovered[(*key, "known", i)], recovered[(*key, "unknown", i)]) for i in seeds]
        only = [sum(k and not u for k, u in pairs), sum(u and not k for k, u in pairs)]
        pooled[b][0] += only[0]
        pooled[b][1] += only[1]
        top[b] = max(top[b], (only[0], f"{n}-qubit ${reps}\\times{layers}$"))
        lines.append(
            f"{n} & ${reps}\\times{layers}$ & {b} & {cells[0]} & {cells[1]} & "
            f"{mcnemar_exact(*only):.3f} \\\\"
        )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")
    known_all = sum(k for k, _ in pooled.values())
    unknown_all = sum(u for _, u in pooled.values())
    macros = {
        "LabelKnownOnlyAll": str(known_all),
        "LabelUnknownOnlyAll": str(unknown_all),
        "LabelPooledPAll": f"{mcnemar_exact(known_all, unknown_all):.3f}",
    }
    for b, (known_only, unknown_only) in sorted(pooled.items()):
        name = _batch_name(b)
        macros[f"LabelKnownOnly{name}"] = str(known_only)
        macros[f"LabelUnknownOnly{name}"] = str(unknown_only)
        macros[f"LabelPooledP{name}"] = f"{mcnemar_exact(known_only, unknown_only):.3f}"
        macros[f"LabelKnownOnlyTop{name}"] = str(top[b][0])
        macros[f"LabelTopCircuit{name}"] = top[b][1]
    return macros


def _batch_name(b: int) -> str:
    """Batch size as a LaTeX-safe macro suffix (macro names cannot hold digits)."""
    return ("One", "Two", "Four", "Eight")[(1, 2, 4, 8).index(b)]


def write_macros(macros: dict[str, str], out: Path) -> None:
    """Numbers quoted in the report prose, as \\newcommand definitions."""
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in sorted(macros.items())))


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
        budget_table(load([budget]), args.report / "tables" / "budget.tex")
    macros: dict[str, str] = {}
    rq3 = args.results / "rq3.jsonl"
    if rq3.exists():
        # Batch 1 of the same circuits is already in RQ1.
        circuits = {(r["n_qubits"], r["reps"], r["layers"]) for r in load([rq3])}
        base = [r for r in rq1 if (r["n_qubits"], r["reps"], r["layers"]) in circuits]
        batch = base + load([rq3])
        batch_map(batch, args.report / "figures" / "batch.pdf")
        macros.update(batch_table(batch, args.report / "tables" / "rq3.tex"))
    rq4 = args.results / "rq4.jsonl"
    if rq4.exists():
        macros.update(label_table(load([rq4]), args.report / "tables" / "rq4.tex"))
    write_macros(macros, args.report / "tables" / "numbers.tex")


if __name__ == "__main__":
    main()
