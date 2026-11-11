"""Figures and tables for the report, drawn from the sweep JSONL files in results/.

uv run python -m qflp.figures

Writes report/figures/*.pdf and report/tables/*.tex. The report inputs the tables, so no number in
them is typed by hand.
"""

import argparse
import itertools
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from qflp import macros as prose
from qflp.summarize import CLOSE_RAD, load, table

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


def privacy_map(rows: list[dict], out: Path, restarts: int) -> None:
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


def budget_table(rows: list[dict], out: Path, budgets: tuple[int, int]) -> None:
    """LaTeX rows: qubits, reps x layers, effective params, seeds recovered within each restart
    budget, and the fraction of all restarts that recover the input."""
    small, large = budgets
    rows = [r for r in rows if r["shots"] == 0 and r["batch"] == 1]
    if min(r["restarts"] for r in rows) < large:
        raise ValueError(f"a budget run has fewer restarts than the {large} budget")
    lines = []
    for (n, reps, layers), s in prose.budget_stats(rows, small).items():
        lines.append(
            f"{n} & ${reps}\\times{layers}$ & {s['n_effective']} & {s['before']}/{s['seeds']} & "
            f"{s['after']}/{s['seeds']} & {100 * s['rate']:.1f}\\% \\\\"
        )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")


def _exact_fixed(rows: list[dict], restarts: int) -> list[dict]:
    return [
        r for r in rows if r["shots"] == 0 and r["label"] == "fixed" and r["restarts"] == restarts
    ]


def batch_map(rows: list[dict], out: Path, restarts: int) -> None:
    """Recovered fraction against unknowns per effective parameter (N·B / P_eff), one line per
    circuit, so circuits of different size share the point where the inputs outnumber the
    equations."""
    rows = _exact_fixed(rows, restarts)
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


def batch_table(rows: list[dict], out: Path, restarts: int) -> dict[str, str]:
    """LaTeX rows: qubits, reps x layers, effective params, batch, unknowns, four outcome counts,
    and the median error of the worst-matched input.

    Returns report macros: the largest unknowns-per-parameter ratio with any recovery, the
    smallest with none, the fewest seeds recovered below a ratio of 1, and the range of median
    errors among settings past it."""
    summary = table(_exact_fixed(rows, restarts))
    ratio = {id(s): s["n_qubits"] * s["batch"] / s["n_effective"] for s in summary}
    leaked = [s for s in summary if s["recovered"] > 0]
    held = [s for s in summary if s["recovered"] == 0]
    if max(ratio[id(s)] for s in leaked) >= min(ratio[id(s)] for s in held):
        raise prose.ClaimError("recovery does not separate on unknowns per effective parameter")
    if not max(ratio[id(s)] for s in leaked) < 1 < min(ratio[id(s)] for s in held):
        raise prose.ClaimError("the grid brackets one unknown per effective parameter")
    errors = [s["median_error"] for s in summary if ratio[id(s)] > 1]
    if any(s["ambiguous"] != s["seeds"] for s in summary if ratio[id(s)] > 1):
        raise prose.ClaimError("past one unknown per parameter, every seed was ambiguous")
    below = [s for s in summary if ratio[id(s)] < 1 and s["recovered"] < s["seeds"]]
    if any(s["close"] for s in below):
        raise prose.ClaimError("below one unknown per parameter, no failure was close")
    if any(s["ambiguous"] for s in below):
        raise prose.ClaimError("below one unknown per parameter, no failure was ambiguous")
    stuck = {}
    for s in below:
        circuit = (s["n_qubits"], s["reps"], s["layers"])
        stuck[circuit] = max(stuck.get(circuit, s), s, key=lambda x: x["stuck"])
    if len({n for n, _, _ in stuck}) != 1:
        raise prose.ClaimError("the stuck seeds below a ratio of 1 sit at one size")
    if len(stuck) != 2:
        raise prose.ClaimError("the stuck seeds below a ratio of 1 sit on two circuits")
    for circuit in stuck:
        mine = [s for s in summary if (s["n_qubits"], s["reps"], s["layers"]) == circuit]
        line = [s["stuck"] for s in sorted(mine, key=lambda s: s["batch"]) if ratio[id(s)] < 1]
        if any(a >= b for a, b in itertools.pairwise(line)):
            raise prose.ClaimError("below a ratio of 1, a larger batch left more seeds stuck")
    (shallow, deep) = (stuck[k] for k in sorted(stuck, key=lambda k: k[2]))
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
        "BatchAmbigErrLow": f"{min(errors):.2f}",
        "BatchAmbigErrHigh": f"{max(errors):.2f}",
        "BatchStuckQubits": str(shallow["n_qubits"]),
        "BatchStuckShallowCircuit": f"${shallow['reps']}\\times{shallow['layers']}$",
        "BatchStuckShallow": str(shallow["stuck"]),
        "BatchStuckShallowAt": str(shallow["batch"]),
        "BatchStuckDeepCircuit": f"${deep['reps']}\\times{deep['layers']}$",
        "BatchStuckDeep": str(deep["stuck"]),
        "BatchStuckDeepAt": str(deep["batch"]),
    }


def _discordant(rows: list[dict], setting: tuple, arm: str, a, b) -> dict[tuple, tuple[int, int]]:
    """Per setting, the seeds recovered only when `arm` is `a` and only when it is `b`, pairing the
    two runs of each seed."""
    rec = {
        (*(r[k] for k in setting), r[arm], r["seed"]): r["best_outcome"] == "recovered"
        for r in rows
    }
    out = {}
    for key in sorted({k[:-2] for k in rec}):
        seeds = sorted({k[-1] for k in rec if k[:-2] == key})
        pairs = [(rec[(*key, a, i)], rec[(*key, b, i)]) for i in seeds]
        out[key] = (sum(x and not y for x, y in pairs), sum(y and not x for x, y in pairs))
    return out


def _outcome_cell(s: dict) -> str:
    return f"{s['recovered']}/{s['close']}/{s['ambiguous']}/{s['stuck']}"


def label_table(rows: list[dict], out: Path) -> dict[str, str]:
    """LaTeX rows: qubits, reps x layers, batch, recovered/close/ambiguous/stuck with the label
    known and with it unknown on the same clients, and the exact McNemar p-value for recovery.

    Returns report macros for the same test pooled over every setting (the primary test) and, as
    description, per batch size; for the batch size with the most known-only seeds, the single
    setting with the largest share of them and its stuck counts. Clients are independent across
    settings, so their discordant pairs add."""
    rows = [r for r in rows if r["shots"] == 0]
    setting = ("n_qubits", "reps", "layers", "batch", "restarts")
    summary = {(*(s[k] for k in setting), s["label"]): s for s in table(rows)}
    lines = []
    pooled: dict[int, list[int]] = defaultdict(lambda: [0, 0])
    p_setting = []
    top: dict[int, tuple[int, str, tuple]] = defaultdict(lambda: (0, "", ()))
    for key, only in _discordant(rows, setting, "label", "known", "unknown").items():
        n, reps, layers, b, _ = key
        cells = [_outcome_cell(summary[(*key, label)]) for label in ("known", "unknown")]
        pooled[b][0] += only[0]
        pooled[b][1] += only[1]
        top[b] = max(top[b], (only[0], f"{n}-qubit ${reps}\\times{layers}$", key))
        p_setting.append(mcnemar_exact(*only))
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
        "LabelPooledPAll": _p_relation(mcnemar_exact(known_all, unknown_all)),
    }
    for b, (known_only, unknown_only) in sorted(pooled.items()):
        name = _batch_name(b)
        macros[f"LabelKnownOnly{name}"] = str(known_only)
        macros[f"LabelUnknownOnly{name}"] = str(unknown_only)
        macros[f"LabelPooledP{name}"] = _p_relation(mcnemar_exact(known_only, unknown_only))
    b = max(pooled, key=lambda b: pooled[b][0])
    known_only, circuit, key = top[b]
    unknown = summary[(*key, "unknown")]
    known_on_top = summary[(*key, "known")]
    if known_all <= unknown_all:
        raise prose.ClaimError("the label-free attack recovered fewer inputs")
    if mcnemar_exact(known_all, unknown_all) >= 0.05:
        raise prose.ClaimError("the pooled label test is significant (p < 0.05)")
    if unknown["stuck"] <= known_on_top["stuck"]:
        raise prose.ClaimError("hiding the label left more seeds stuck on that circuit")
    if b != max(pooled):
        raise prose.ClaimError("the label difference sits at the largest batch")
    if 2 * known_only <= pooled[b][0]:
        raise prose.ClaimError("the label difference sits mostly on one circuit")
    if unknown["ambiguous"]:
        raise prose.ClaimError("the label-free failures on that circuit were not ambiguous")
    if unknown["close"]:
        raise prose.ClaimError("the label-free failures on that circuit were not close")
    if min(p_setting) < 0.05:
        raise prose.ClaimError("no single setting's difference is significant")
    macros |= {
        "LabelTopBatch": str(b),
        "LabelKnownOnlyTop": str(known_only),
        "LabelTopCircuit": circuit,
        "LabelTopStuckKnown": str(summary[(*key, "known")]["stuck"]),
        "LabelTopStuckUnknown": str(summary[(*key, "unknown")]["stuck"]),
    }
    return macros


TASK_LABEL = {"linear": "Linear", "curves2": "Curves, $D{=}2$", "curves10": "Curves, $D{=}10$"}
TASK_MACRO = {"linear": "Linear", "curves2": "CurvesTwo", "curves10": "CurvesTen"}
TASK_PROSE = {"linear": "the linear task", "curves2": "curves with $D = 2$",
              "curves10": "curves with $D = 10$"}  # fmt: skip
# The task the report discusses as the lead: the one with the smallest per-task p.
LEAD_TASK = "curves2"


def _p_relation(p: float) -> str:
    """A p-value as math text that follows `$p$`: `= 0.627`, two significant figures below 0.01
    (`= 0.0015`), `< 0.001` below 0.0005, and `< 0.05` where three decimals would round a p below
    0.05 up to 0.050."""
    if p < 0.0005:
        return "< 0.001"
    if p < 0.01:
        return f"= {p:.2g}"
    if p < 0.05 <= round(p, 3):
        return "< 0.05"
    return f"= {p:.3f}"


def trained_map(rows: list[dict], out: Path) -> None:
    """Recovered fraction against training step for the one circuit that varies, one line per task
    and one panel per qubit count."""
    rows = [r for r in rows if (r["reps"], r["layers"]) == (4, 1)]
    steps = sorted({r["step"] for r in rows})
    qubits = sorted({r["n_qubits"] for r in rows})
    tasks = [t for t in TASK_LABEL if any(r["task"] == t for r in rows)]
    fig, axes = plt.subplots(1, len(qubits), figsize=(3.5 * len(qubits), 2.9), sharey=True)
    axes = [axes] if len(qubits) == 1 else list(axes)
    for ax, n in zip(axes, qubits, strict=True):
        for i, (task, (colour, marker)) in enumerate(zip(tasks, _styles(len(tasks)), strict=True)):
            stats = [
                _recovered_with_interval(
                    [
                        r["best_outcome"] == "recovered"
                        for r in rows
                        if (r["n_qubits"], r["task"], r["step"]) == (n, task, st)
                    ]
                )
                for st in steps
            ]
            ax.errorbar(
                [j + 0.06 * (i - 1) for j in range(len(steps))],  # separate overlapping bars
                [f for f, _, _ in stats],
                yerr=[[d for _, d, _ in stats], [u for _, _, u in stats]],
                color=colour,
                marker=marker,
                ms=6,
                lw=2,
                capsize=3,
                label=TASK_LABEL[task],
            )
        ax.set_xticks(range(len(steps)), labels=[str(st) for st in steps])
        ax.set_xlabel("training steps before the attack")
        ax.set_title(f"{n} qubits", fontsize=10)
        ax.set_ylim(-0.03, 1.03)
        _tidy(ax)
        ax.legend(frameon=False, fontsize=8, loc="lower left")
    axes[0].set_ylabel("seeds recovered (fraction)")
    _save(fig, out)


def trained_table(rows: list[dict], baselines: list[dict], out: Path) -> dict[str, str]:
    """LaTeX rows: task, qubits, reps x layers, median test accuracy of the trained circuit and the
    logistic-regression test accuracy, then recovered/close/ambiguous/stuck at each training step.

    Returns report macros for the paired test of recovery before and after training, pooled over
    every setting (the primary test) and per task, and the median gradient norm ratio."""
    steps = sorted({r["step"] for r in rows})
    first, last = steps[0], steps[-1]
    setting = ("task", "n_qubits", "reps", "layers")
    summary = {(*(s[k] for k in setting), s["step"]): s for s in table(rows)}
    logistic = {(b["task"], b["n_qubits"]): b["test_acc"] for b in baselines}
    order = {t: i for i, t in enumerate(TASK_LABEL)}
    lines = []
    gap: dict[tuple, float] = {}
    accs: dict[str, list[float]] = defaultdict(list)
    pooled: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    discordant = _discordant(rows, setting, "step", first, last)
    for key in sorted(discordant, key=lambda k: (order[k[0]], *k[1:])):
        task, n, reps, layers = key
        acc = statistics.median(
            r["test_acc"] for r in rows if (*(r[k] for k in setting), r["step"]) == (*key, last)
        )
        cells = " & ".join(_outcome_cell(summary[(*key, st)]) for st in steps)
        lines.append(
            f"{TASK_LABEL[task]} & {n} & ${reps}\\times{layers}$ & "
            f"{acc:.2f} ({logistic[(task, n)]:.2f}) & {cells} \\\\"
        )
        gap[key] = logistic[(task, n)] - acc
        accs[task] += [acc, logistic[(task, n)]]
        pooled[task][0] += discordant[key][0]
        pooled[task][1] += discordant[key][1]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")
    before = sum(a for a, _ in pooled.values())
    after = sum(b for _, b in pooled.values())
    by_seed = {(*(r[k] for k in setting), r["seed"], r["step"]): r["grad_norm2"] for r in rows}
    ratios = [
        by_seed[(*k[:-1], last)] / by_seed[(*k[:-1], first)] for k in by_seed if k[-1] == first
    ]
    shallow = {k: s["recovered"] for k, s in summary.items() if s["reps"] == 1}
    lowest = min(shallow, key=lambda k: shallow[k])
    seeds = max(s["seeds"] for s in summary.values())
    if 2 * shallow[lowest] <= seeds:
        raise prose.ClaimError("every 1 x l setting was recovered on most seeds at every step")
    if sum(v == shallow[lowest] for v in shallow.values()) > 1:
        raise prose.ClaimError("one setting holds the fewest 1 x l recoveries")
    worst = max(gap, key=lambda k: gap[k])
    if any(r["best_outcome"] not in ("recovered", "stuck") for r in rows):
        raise prose.ClaimError("every trained attack ended recovered or stuck")
    macros = {
        "TrainedOnlyBeforeAll": str(before),
        "TrainedOnlyAfterAll": str(after),
        "TrainedPooledPAll": _p_relation(mcnemar_exact(before, after)),
        "TrainedFinalStep": str(last),
        "TrainedGradRatioMedian": f"{statistics.median(ratios):.2f}",
        "TrainedShallowMinRecovered": str(shallow[lowest]),
        "TrainedAttacks": str(len(rows)),
        "TrainedSteps": prose._and(str(st) for st in steps),
        "TrainedTasks": str(len(pooled)),
        "TrainedWorstGap": f"{gap[worst]:.2f}",
        "TrainedWorstCircuit": f"{worst[1]}-qubit ${worst[2]}\\times{worst[3]}$",
        "TrainedWorstTask": TASK_PROSE[worst[0]],
        "TrainedTenAccLow": f"{min(accs['curves10'], default=0):.2f}",
        "TrainedTenAccHigh": f"{max(accs['curves10'], default=0):.2f}",
    }
    p_task = {task: mcnemar_exact(*counts) for task, counts in pooled.items()}
    for task, (b, a) in pooled.items():
        macros[f"TrainedOnlyBefore{TASK_MACRO[task]}"] = str(b)
        macros[f"TrainedOnlyAfter{TASK_MACRO[task]}"] = str(a)
        macros[f"TrainedPooledP{TASK_MACRO[task]}"] = _p_relation(p_task[task])
    # The report names one task as the lead; Bonferroni over the per-task tests, which are
    # secondary to the pooled one.
    if LEAD_TASK in p_task:
        if min(p_task, key=lambda t: p_task[t]) != LEAD_TASK:
            raise prose.ClaimError(f"{LEAD_TASK} has the smallest per-task p")
        if pooled[LEAD_TASK][0] <= pooled[LEAD_TASK][1]:
            raise prose.ClaimError(f"fewer {LEAD_TASK} seeds were recovered after training")
        if any(p < 0.05 for t, p in p_task.items() if t != LEAD_TASK):
            raise prose.ClaimError("the other per-task tests are not significant")
        p_lead = min(1.0, len(p_task) * p_task[LEAD_TASK])
        macros["TrainedLeadBonferroni"] = _p_relation(p_lead)
        if lowest[0] != LEAD_TASK:
            raise prose.ClaimError(f"the fewest 1 x l recoveries are on {LEAD_TASK}")
    if mcnemar_exact(before, after) < 0.05:
        raise prose.ClaimError("training showed no overall change in recovery (pooled p >= 0.05)")
    if statistics.median(ratios) < 1:
        raise prose.ClaimError("training did not shrink the median gradient")
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
    ap.add_argument("--spike", type=Path, default=Path("spike/results-2026-09-24.txt"))
    args = ap.parse_args(argv)
    rq1 = load([args.results / "rq1.jsonl"])
    rq2 = load([args.results / "rq2.jsonl"])
    budget_rows = load([args.results / "budget.jsonl"])
    files = {
        "rq1": rq1,
        "rq2": rq2,
        "budget": budget_rows,
        "rq3": load([args.results / "rq3.jsonl"]),
        "rq4": load([args.results / "rq4.jsonl"]),
        "rq5": load([args.results / "rq5.jsonl"]),
    }
    outcome_table(rq1, args.report / "tables" / "rq1.tex")
    shot_noise(rq2, args.report / "figures" / "shot-noise.pdf")
    shots_table(rq2, args.report / "tables" / "rq2.tex")
    macros: dict[str, str] = {
        **prose.design_macros(),
        **prose.count_macros(files),
        **prose.rq1_macros(rq1),
        **prose.rq2_macros(rq2, rq1),
        **prose.budget_macros(budget_rows, rq1),
        **prose.spike_macros(args.spike),
        **prose.compile_macros(args.results / "compile-time.jsonl"),
    }
    restarts = int(macros["Restarts"])
    privacy_map(rq1, args.report / "figures" / "privacy-map.pdf", restarts)
    budget_table(
        budget_rows,
        args.report / "tables" / "budget.tex",
        (restarts, int(macros["BudgetRestarts"])),
    )
    # Batch 1 of the same circuits is already in RQ1.
    circuits = {(r["n_qubits"], r["reps"], r["layers"]) for r in files["rq3"]}
    batch = [r for r in rq1 if (r["n_qubits"], r["reps"], r["layers"]) in circuits] + files["rq3"]
    batch_map(batch, args.report / "figures" / "batch.pdf", restarts)
    macros.update(batch_table(batch, args.report / "tables" / "rq3.tex", restarts))
    macros.update(label_table(files["rq4"], args.report / "tables" / "rq4.tex"))
    training = args.results / "rq5-training.jsonl"
    baselines = [r for r in map(json.loads, training.read_text().splitlines()) if "model" in r]
    trained = files["rq5"]
    trained_map(trained, args.report / "figures" / "trained.pdf")
    macros.update(trained_table(trained, baselines, args.report / "tables" / "rq5.tex"))
    write_macros(macros, args.report / "tables" / "numbers.tex")


if __name__ == "__main__":
    main()
