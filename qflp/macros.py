"""Numbers the report's prose quotes for the method, RQ1, RQ2 and the restart budget, as macros.

Each function returns {macro name: LaTeX text}; `qflp.figures` writes them to
report/tables/numbers.tex. A claim the prose makes without a number ("no attack with r >= 2 ended
ambiguous") is checked here and raises `ClaimError` once the results stop supporting it.
"""

import json
import math
import re
import statistics
from collections import defaultdict
from pathlib import Path

from qflp import attack, data, train
from qflp.circuit import REFERENCE_TOL
from qflp.summarize import best_outcome, outcomes, table


class ClaimError(ValueError):
    """A sentence of the report no longer holds on the results."""


def _require(ok: bool, claim: str) -> None:
    if not ok:
        raise ClaimError(claim)


def _and(items) -> str:
    items = list(items)
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


def _circuit(reps: int, layers: int) -> str:
    return f"${reps}\\times{layers}$"


def _pow10(x: float) -> str:
    e = round(math.log10(x))
    if not math.isclose(x, 10.0**e):
        raise ValueError(f"{x} is not a power of ten")
    return f"\\ensuremath{{10^{{{e}}}}}"


def _upper_sci(x: float) -> str:
    """x rounded up to one significant figure, as m x 10^e: a bound that holds for x."""
    e = math.floor(math.log10(x))
    m = math.ceil(x / 10.0**e)
    if m == 10:
        m, e = 1, e + 1
    return f"\\ensuremath{{{m}\\times10^{{{e}}}}}" if m > 1 else f"\\ensuremath{{10^{{{e}}}}}"


def _one(values, what: str):
    found = set(values)
    if len(found) != 1:
        raise ClaimError(f"expected one {what}, found {sorted(found)}")
    return found.pop()


def _sizes(rows: list[dict]) -> tuple[int, int]:
    """The small and the large qubit count of a two-size experiment."""
    qubits = sorted({r["n_qubits"] for r in rows})
    _require(len(qubits) == 2, f"two circuit sizes, found {qubits}")
    return qubits[0], qubits[1]


def design_macros() -> dict[str, str]:
    """Thresholds, budgets and benchmark settings, read from the code that uses them."""
    return {
        "RecoveredRad": f"{attack.RECOVERED_RAD:g}",
        "CloseRad": f"{attack.CLOSE_RAD:g}",
        "ExplainsSlack": _pow10(attack.EXPLAINS_SLACK),
        "RidgeScale": _pow10(attack.RIDGE),
        "MaxIter": str(attack.MAXITER),
        "ReferenceTol": _pow10(REFERENCE_TOL),
        "BenchSamples": str(data.N_SAMPLES),
        "BenchMargin": f"{data.MARGIN_PER_FEATURE:g}",
        "BenchNoise": f"{data.CURVE_NOISE:g}",
        "TrainPercent": str(round(100 * (1 - data.TEST_SIZE))),
        "TestPercent": str(round(100 * data.TEST_SIZE)),
        "ValidationPercent": str(round(100 * train.VALIDATION)),
        "TrainSteps": str(train.STEPS),
        "LearningRates": _and(_pow10(lr) for lr in train.LEARNING_RATES),
        "TuningSeeds": str(train.TUNING_SEEDS),
    }


def count_macros(files: dict[str, list[dict]]) -> dict[str, str]:
    """Seeds per setting and restarts per attack, checked to be the same everywhere they are
    quoted, and the number of attacks in each experiment."""
    seeds = _one((s["seeds"] for rows in files.values() for s in table(rows)), "seed count")
    return {
        "Seeds": str(seeds),
        "Restarts": str(_one((r["restarts"] for r in files["rq1"]), "RQ1 restart count")),
        "BudgetRestarts": str(_one((r["restarts"] for r in files["budget"]), "budget restarts")),
        "RqOneAttacks": str(len(files["rq1"])),
        "RqOneShapes": str(len(table(files["rq1"]))),
        "RqTwoAttacks": str(len(files["rq2"])),
        "BatchLabelAttacks": str(len(files["rq3"]) + len(files["rq4"])),
    }


def rq1_macros(rows: list[dict]) -> dict[str, str]:
    """RQ1: the three regimes of encoding repetitions against trainable layers."""
    small, large = _sizes(rows)
    by = {(s["n_qubits"], s["reps"], s["layers"]): s for s in table(rows)}
    seeds = _one((s["seeds"] for s in by.values()), "seed count")
    out: dict[str, str] = {}

    # r = l = 1: as many effective parameters as inputs.
    for n, name in ((small, "Small"), (large, "Large")):
        square = [r for r in rows if (r["n_qubits"], r["reps"], r["layers"]) == (n, 1, 1)]
        _require(
            all(r["best_match"] <= attack.EXPLAINS_SLACK * r["grad_norm2"] for r in square),
            "the best restart matched the gradient exactly on every r = l = 1 seed",
        )
        s = by[(n, 1, 1)]
        out[f"SquareWrong{name}"] = str(seeds - s["recovered"])
        out[f"SquareOtherRestart{name}"] = str(
            sum(r["any_recovered"] and r["best_outcome"] != "recovered" for r in square)
        )
    close = [
        r["best_error"]
        for r in rows
        if r["reps"] == r["layers"] == 1 and r["best_outcome"] == "close"
    ]
    _require(by[(large, 1, 1)]["close"] == 0, f"no close r = l = 1 seed at {large} qubits")
    out["SquareCloseSmall"] = str(by[(small, 1, 1)]["close"])
    out["SquareCloseMaxErr"] = f"{max(close):.2f}"

    # One encoding layer, two or more trainable layers.
    shallow = [s for (_, reps, layers), s in by.items() if reps == 1 and layers >= 2]
    _require(all(s["recovered"] == seeds for s in shallow), "every 1 x l >= 2 circuit recovered")
    out["ShallowMinParams"] = str(min(s["n_effective"] for s in shallow))
    out["ShallowMaxParams"] = str(max(s["n_effective"] for s in shallow))

    # The largest fixed product of reps and layers, one count per number of repetitions.
    top = max(reps * layers for _, reps, layers in by)
    for n, name in ((small, "Small"), (large, "Large")):
        line = sorted((k[1], s) for k, s in by.items() if k[0] == n and k[1] * k[2] == top)
        _require([reps for reps, _ in line] == [1, 2, 4, 8], "r = 1, 2, 4, 8 at the top product")
        out[f"RecByReps{name}"] = _and(str(s["recovered"]) for _, s in line)
        out[f"ParamsTop{name}"] = str(_one((s["n_effective"] for _, s in line), "parameter count"))

    deep = [(reps, layers) for (n, reps, layers) in by if n == small and reps >= 2]
    _require(
        all(by[(large, *g)]["stuck"] > by[(small, *g)]["stuck"] for g in deep),
        "every r >= 2 shape was stuck on more seeds at the larger size",
    )
    _require(
        all(s["ambiguous"] == 0 for (_, reps, _), s in by.items() if reps >= 2),
        "no attack with r >= 2 ended ambiguous",
    )
    out["StuckFourOneSmall"] = str(by[(small, 4, 1)]["stuck"])
    out["StuckFourOneLarge"] = str(by[(large, 4, 1)]["stuck"])
    return out


def _slope(points: list[tuple[float, float]]) -> float:
    """Least-squares slope of log10(y) against log10(x)."""
    xs = [math.log10(x) for x, _ in points]
    ys = [math.log10(y) for _, y in points]
    mx, my = statistics.fmean(xs), statistics.fmean(ys)
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys, strict=True)) / sum(
        (x - mx) ** 2 for x in xs
    )


def rq2_macros(rows: list[dict], rq1: list[dict]) -> dict[str, str]:
    """RQ2: reconstruction error under finite-shot gradients."""
    small, large = _sizes(rows)
    by = {(s["n_qubits"], s["reps"], s["layers"], s["shots"]): s for s in table(rows)}
    seeds = _one((s["seeds"] for s in by.values()), "seed count")
    shots = sorted({k[3] for k in by})
    circuits = sorted({k[:3] for k in by})
    low, second = shots[0], shots[1]
    out: dict[str, str] = {}
    for n, name in ((small, "Small"), (large, "Large")):
        out[f"ShotParams{name}"] = str(
            _one((s["n_effective"] for k, s in by.items() if k[0] == n), "parameter count")
        )
    at_low = {n: [by[(*c, low)] for c in circuits if c[0] == n] for n in (small, large)}
    out["ShotLowErrSmallMin"] = f"{min(s['median_error'] for s in at_low[small]):.3f}"
    out["ShotLowErrSmallMax"] = f"{max(s['median_error'] for s in at_low[small]):.3f}"
    out["ShotLowWithinSmallMin"] = str(min(s["within_close"] for s in at_low[small]))
    out["ShotLowWithinSmallMax"] = str(max(s["within_close"] for s in at_low[small]))
    out["ShotLowErrLargeMin"] = f"{min(s['median_error'] for s in at_low[large]):.3f}"
    out["ShotLowErrLargeMax"] = f"{max(s['median_error'] for s in at_low[large]):.3f}"
    out["ShotLowWithinLargeMax"] = str(max(s["within_close"] for s in at_low[large]))
    _require(
        all(2 * s["within_close"] < seeds for s in at_low[large]),
        f"at {large} qubits and {low} shots every circuit was private on most seeds",
    )
    _require(
        all(2 * s["within_close"] >= seeds for k, s in by.items() if k[0] == small),
        f"at {small} qubits every circuit leaked on at least half the seeds at every shot count",
    )
    private_second = [
        c for c in circuits if c[0] == large and 2 * by[(*c, second)]["within_close"] < seeds
    ]
    out["ShotPrivateAtSecond"] = _and(_circuit(c[1], c[2]) for c in private_second)

    # Error against shots, where the attack lands within CLOSE_RAD; shot noise alone gives -1/2.
    slopes = []
    for c in circuits:
        pts = [(s, by[(*c, s)]["median_error"]) for s in shots]
        pts = [(s, e) for s, e in pts if e <= attack.CLOSE_RAD]
        if len(pts) >= 3:
            slopes.append(_slope(pts))
    out["ShotSlopeCircuits"] = str(len(slopes))
    out["ShotSlopeMin"] = f"\\ensuremath{{{min(slopes):.2f}}}"
    out["ShotSlopeMax"] = f"\\ensuremath{{{max(slopes):.2f}}}"

    wide = (large, 1, 4)
    full = [s for s in shots if by[(*wide, s)]["within_close"] == seeds]
    out["ShotWideAllAt"] = _pow10(min(full))

    deep = (large, 4, 1)
    _require(
        all(2 * by[(*deep, s)]["within_close"] < seeds for s in shots),
        f"the {large}-qubit 4 x 1 circuit stayed private on most seeds at every shot count",
    )
    clean = [s for s in shots if by[(*deep, s)]["ambiguous"] == 0]
    _require(clean == shots[shots.index(clean[0]) :], "no ambiguous seed from that shot count up")
    stuck = [by[(*deep, s)]["stuck"] for s in clean]
    out["ShotDeepStuckFrom"] = _pow10(clean[0])
    out["ShotDeepStuckMin"] = str(min(stuck))
    out["ShotDeepStuckMax"] = str(max(stuck))

    def stuck_seeds(rs, shot):
        return {r["seed"] for r in rs if r["shots"] == shot and r["best_outcome"] == "stuck"}

    same = [r for r in rows if (r["n_qubits"], r["reps"], r["layers"]) == (small, 4, 1)]
    exact = [r for r in rq1 if (r["n_qubits"], r["reps"], r["layers"]) == (small, 4, 1)]
    _require(
        all(stuck_seeds(same, s) == stuck_seeds(exact, 0) for s in shots),
        f"the {small}-qubit 4 x 1 circuit was stuck on the same seeds at every shot count",
    )
    return out


def budget_macros(rows: list[dict], rq1: list[dict]) -> dict[str, str]:
    """The restart budget: the RQ1 circuits with the most stuck seeds, rerun with more restarts."""
    small_budget = _one((r["restarts"] for r in rq1), "RQ1 restart count")
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        groups[(r["n_qubits"], r["reps"], r["layers"])].append(r)
    base = {(r["n_qubits"], r["reps"], r["layers"], r["seed"]): r for r in rq1}
    for key, rs in groups.items():
        for r in rs:
            first = [d["error"] for d in r["restarts_detail"][:small_budget]]
            again = [d["error"] for d in base[(*key, r["seed"])]["restarts_detail"]]
            _require(
                all(
                    math.isclose(a, b, rel_tol=1e-6, abs_tol=1e-9)
                    for a, b in zip(first, again, strict=True)
                ),
                "the first restarts of each budget run repeat the RQ1 attack",
            )
    seeds = _one((len(rs) for rs in groups.values()), "seed count")
    stats = {}
    for key, rs in groups.items():
        per = [o == "recovered" for r in rs for o in outcomes(r)]
        stats[key] = {
            "before": sum(best_outcome(r, small_budget) == "recovered" for r in rs),
            "after": sum(r["best_outcome"] == "recovered" for r in rs),
            "rate": sum(per) / len(per),
        }
    hardest = min(stats, key=lambda k: stats[k]["rate"])
    easiest = max(stats, key=lambda k: stats[k]["rate"])
    others = sorted(k for k in stats if k != hardest)
    _require(2 * stats[hardest]["after"] < seeds, "the hardest circuit held on most seeds")
    _require(all(2 * stats[k]["after"] > seeds for k in others), "the others leaked on most")
    seconds = statistics.median(r["seconds"] / r["restarts"] for r in rows)
    per_success = round(1 / stats[hardest]["rate"])
    errors = [
        min(r["restarts_detail"], key=lambda d: d["match"])["error"]
        for r in rows
        if r["best_outcome"] == "recovered"
    ]
    return {
        "BudgetCircuits": str(len(stats)),
        "BudgetOthersCount": str(len(others)),
        "BudgetRises": _and(
            f"{stats[k]['before']} to {stats[k]['after']} ({_circuit(k[1], k[2])})" for k in others
        ),
        "BudgetOthersMin": str(min(stats[k]["after"] for k in others)),
        "BudgetOthersMax": str(max(stats[k]["after"] for k in others)),
        "BudgetHardestCircuit": _circuit(hardest[1], hardest[2]),
        "BudgetHardestRecovered": str(stats[hardest]["after"]),
        "BudgetHardestHeld": str(seeds - stats[hardest]["after"]),
        "BudgetHardestRate": f"{100 * stats[hardest]['rate']:.1f}",
        "BudgetHardestPerSuccess": str(per_success),
        "BudgetEasiestCircuit": _circuit(easiest[1], easiest[2]),
        "BudgetEasiestRate": f"{100 * stats[easiest]['rate']:.1f}",
        "BudgetEasiestPerSuccess": str(round(1 / stats[easiest]["rate"])),
        "BudgetSecondsPerRestart": f"{seconds:.2f}",
        "BudgetHardestSeconds": str(round(per_success * seconds)),
        "BudgetMaxRecoveredErr": _upper_sci(max(errors)),
    }


def spike_macros(path: Path) -> dict[str, str]:
    """The PennyLane feasibility spike's cost for its deepest circuit, from its committed log."""
    text = path.read_text()
    head = re.search(r"\((\d+) qubits.*?(\d+) seeds x (\d+) L-BFGS-B restarts", text, re.S)
    lines = re.findall(r"^reps=(\d+) layers=(\d+) .* (\d+)s$", text, re.M)
    if head is None or not lines:
        raise ValueError(f"{path}: not the spike log")
    reps, layers, seconds = max(lines, key=lambda t: int(t[0]) * int(t[1]))
    return {
        "SpikeQubits": head[1],
        "SpikeSeeds": head[2],
        "SpikeRestarts": head[3],
        "SpikeReps": reps,
        "SpikeLayers": layers,
        "SpikeSeconds": seconds,
    }


def compile_macros(path: Path) -> dict[str, str]:
    """First-call (compiling) time of the JAX attack, from `qflp.compile_time --out`."""
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    return {
        "CompileQubits": str(_one((r["n_qubits"] for r in rows), "qubit count")),
        "CompileShapes": _and(_circuit(r["reps"], r["layers"]) for r in rows),
        "CompileMaxSeconds": f"{max(r['first_call_s'] for r in rows):.1f}",
        "CompileCpus": str(_one((r["cpus"] for r in rows), "CPU count")),
    }
