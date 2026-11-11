"""The report's generated tables and prose numbers match results/, and every claim guard fires."""

import copy
import json
import re
from pathlib import Path

import pytest

from qflp import macros
from qflp.figures import batch_table, budget_table, label_table, trained_table
from qflp.figures import main as figures_main
from qflp.summarize import best_outcome, load, outcomes

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "report"
RESULTS = ROOT / "results"


@pytest.fixture(scope="module")
def regenerated(tmp_path_factory) -> Path:
    out = tmp_path_factory.mktemp("report")
    figures_main(
        [
            "--results",
            str(RESULTS),
            "--report",
            str(out),
            "--spike",
            str(ROOT / "spike" / "results-2026-09-24.txt"),
        ]
    )
    return out


def test_committed_tables_and_numbers_match_the_results(regenerated):
    committed = sorted(p.name for p in (REPORT / "tables").glob("*.tex"))
    assert committed == sorted(p.name for p in (regenerated / "tables").glob("*.tex"))
    for name in committed:
        assert (REPORT / "tables" / name).read_text() == (
            regenerated / "tables" / name
        ).read_text(), f"report/tables/{name} is stale: run `uv run python -m qflp.figures`"


def test_every_generated_number_is_quoted_and_every_quoted_one_is_generated():
    defined = set(
        re.findall(r"\\newcommand\{\\(\w+)\}", (REPORT / "tables/numbers.tex").read_text())
    )
    source = re.sub(r"(?<!\\)%.*", "", (REPORT / "report.tex").read_text())  # drop comments
    used = set(re.findall(r"\\([A-Z]\w*)(?![A-Za-z])", source))
    assert defined - used == set(), "generated but never quoted"
    # Capitalised commands the report uses that are neither generated nor LaTeX's own.
    latex = {"IEEEauthorblockN", "IEEEauthorblockA", "IEEEkeywords"}
    assert used - defined - latex == set(), "quoted but not generated"


@pytest.mark.parametrize(
    ("x", "text"),
    [
        (1.93e-6, "\\ensuremath{2\\times10^{-6}}"),
        (5e-6, "\\ensuremath{5\\times10^{-6}}"),
        (1e-6, "\\ensuremath{10^{-6}}"),
        (9.5e-3, "\\ensuremath{10^{-2}}"),
        (3.0, "\\ensuremath{3\\times10^{0}}"),
    ],
)
def test_upper_sci_rounds_up_to_one_figure(x, text):
    assert macros._upper_sci(x) == text


@pytest.mark.parametrize(
    ("x", "digits", "text"), [(0.1355, 2, "0.14"), (0.1449, 2, "0.15"), (2.04, 1, "2.1"),
                              (0.14, 2, "0.14")]
)  # fmt: skip
def test_ceil_is_an_upper_bound(x, digits, text):
    assert macros._ceil(x, digits) == text


def test_pow10_refuses_a_value_that_is_not_a_power_of_ten():
    assert macros._pow10(1e-3) == "\\ensuremath{10^{-3}}"
    with pytest.raises(ValueError, match="power of ten"):
        macros._pow10(2e-3)


def test_one_and_sizes_refuse_mixed_values():
    with pytest.raises(macros.ClaimError, match="expected one"):
        macros._one([1, 2], "x")
    with pytest.raises(macros.ClaimError, match="two circuit sizes"):
        macros._sizes([{"n_qubits": n} for n in (2, 4, 8)])


@pytest.fixture(scope="module")
def real():
    rows = {
        k: load([RESULTS / f"{k}.jsonl"]) for k in ("rq1", "rq2", "budget", "rq3", "rq4", "rq5")
    }
    circuits = {(r["n_qubits"], r["reps"], r["layers"]) for r in rows["rq3"]}
    rows["batch"] = [
        r for r in rows["rq1"] if (r["n_qubits"], r["reps"], r["layers"]) in circuits
    ] + rows["rq3"]
    training = (RESULTS / "rq5-training.jsonl").read_text().splitlines()
    rows["baselines"] = [r for r in map(json.loads, training) if "model" in r]
    return rows


def _set(rows, match, every=False, **fields):
    """A copy of rows with `fields` set on the first row matching `match`, or on every one."""
    rows = copy.deepcopy(rows)
    hits = [r for r in rows if all(r[k] == v for k, v in match.items())]
    assert hits, match
    for r in hits if every else hits[:1]:
        r.update(fields)
    return rows


def _drop(rows, match):
    return [r for r in rows if not all(r[k] == v for k, v in match.items())]


def _rq1(mutant, d):
    macros.rq1_macros(mutant)


def _rq2(mutant, d):
    macros.rq2_macros(mutant, d["rq1"])


def _budget(mutant, d):
    macros.budget_macros(mutant, d["rq1"])


def _budget_rq1(mutant, d):
    macros.budget_macros(d["budget"], mutant)


def _batch(mutant, d):
    batch_table(mutant, d["tmp"] / "rq3.tex", 10)


def _label(mutant, d):
    label_table(mutant, d["tmp"] / "rq4.tex")


def _trained(mutant, d):
    trained_table(mutant, d["baselines"], d["tmp"] / "rq5.tex")


def _rescore(rows):
    """Recompute each row's best outcome from its restarts, as `load` does."""
    for r in rows:
        r["best_outcome"] = best_outcome(r)
        r["best_error"] = min(r["restarts_detail"], key=lambda d: d["match"])["error"]
    return rows


def _key(r):
    return (r["n_qubits"], r["reps"], r["layers"])


def _miss_after(rows, circuit, start):
    """Every restart of `circuit` from index `start` on misses the input."""
    rows = copy.deepcopy(rows)
    for r in rows:
        if _key(r) == circuit:
            for x in r["restarts_detail"][start:]:
                x["error"] = 3.0
    return rows


def _rate_120(d):
    """Every restart of the 8-qubit 4 x 2 circuit misses, in RQ1 and in the budget rerun alike."""
    rq1, budget = _miss_after(d["rq1"], (8, 4, 2), 0), _miss_after(d["budget"], (8, 4, 2), 0)
    for rows in (rq1, budget):
        _rescore([r for r in rows if _key(r) == (8, 4, 2)])
    macros.budget_macros(budget, rq1)


def _first_seed_not_best(d):
    """A restart that was not the best moves, so the best outcome stays."""
    budget = copy.deepcopy(d["budget"])
    r = next(r for r in budget if r["best_outcome"] == "stuck")
    first = r["restarts_detail"][:10]
    best = min(range(10), key=lambda i: first[i]["match"])
    first[(best + 1) % 10]["error"] += 0.5
    _budget(budget, d)


def _rq1_outcome_flipped(d):
    """RQ1's stored outcome for a budget seed differs from what its restarts give."""
    rq1 = copy.deepcopy(d["rq1"])
    r = next(r for r in rq1 if _key(r) == (8, 2, 1) and r["best_outcome"] == "stuck")
    r["best_outcome"] = "recovered"
    macros.budget_macros(d["budget"], rq1)


def _swap(rows, field, a, b, keep=lambda r: False):
    rows = copy.deepcopy(rows)
    for r in rows:
        if not keep(r) and r[field] in (a, b):
            r[field] = b if r[field] == a else a
    return rows


def _trained_swap(d, tasks):
    _trained(_swap(d["rq5"], "step", 0, 300, keep=lambda r: r["task"] not in tasks), d)


def _linear_significant(d):
    """Linear gains 8 seeds after training and loses none, as a significant but weaker shift."""
    rq5 = copy.deepcopy(d["rq5"])
    start = {(r["n_qubits"], r["reps"], r["layers"], r["seed"]): r["best_outcome"] for r in rq5
             if r["task"] == "linear" and r["step"] == 0}  # fmt: skip
    gained = 0
    for r in rq5:
        if r["task"] == "linear" and r["step"] == 300:
            was = start[(r["n_qubits"], r["reps"], r["layers"], r["seed"])]
            if was == "stuck" and gained < 8:
                r["best_outcome"], gained = "recovered", gained + 1
            else:
                r["best_outcome"] = was
    _trained(rq5, d)


def _shrunk_gradient(d):
    rq5 = copy.deepcopy(d["rq5"])
    for r in rq5:
        if r["step"] == 300:
            r["grad_norm2"] *= 0.1
    _trained(rq5, d)


def _label_rows(d, setting, label, outcome_from, outcome_to, n):
    rows = copy.deepcopy(d["rq4"])
    hit = [r for r in rows if (*_key(r), r["batch"], r["label"]) == (*setting, label)
           and r["best_outcome"] == outcome_from]  # fmt: skip
    assert len(hit) >= n
    for r in hit[:n]:
        r["best_outcome"] = outcome_to
    return rows


def _significant_setting(d):
    """Every 8-qubit 1 x 2, B = 2 seed is recovered with the label and stuck without it."""
    rq4 = _set(d["rq4"], {"n_qubits": 8, "reps": 1, "layers": 2, "batch": 2, "label": "known"},
               True, best_outcome="recovered")  # fmt: skip
    rq4 = _set(rq4, {"n_qubits": 8, "reps": 1, "layers": 2, "batch": 2, "label": "unknown"},
               True, best_outcome="stuck")  # fmt: skip
    _label(rq4, d)


def _no_slope(d):
    """No circuit's median error reaches CLOSE_RAD, while every other RQ2 claim still holds."""
    rq2 = copy.deepcopy(d["rq2"])
    for r in rq2:
        far = r["n_qubits"] == 8 and (r["shots"] == 100 or r["reps"] == 4)
        r["best_error"] = 0.5 if far or r["seed"] >= 10 else 0.1
    _rq2(rq2, d)


def _batch_stuck_falls(d):
    """The 8-qubit 1 x 4 circuit is stuck on 9 seeds at B = 2, more than its 8 at B = 4."""
    rows = copy.deepcopy(d["batch"])
    hit = [r for r in rows if (*_key(r), r["batch"]) == (8, 1, 4, 2)
           and r["best_outcome"] == "recovered"]  # fmt: skip
    for r in hit[:8]:
        r["best_outcome"] = "stuck"
    _batch(rows, d)


def _stuck_not_recovered(d):
    """Six 2 x 1 seeds that 50 restarts recovered stay stuck instead."""
    budget = copy.deepcopy(d["budget"])
    hit = [r for r in budget if _key(r) == (8, 2, 1) and r["best_outcome"] == "recovered"
           and best_outcome(r, 10) == "stuck"]  # fmt: skip
    for r in hit[:6]:
        r["best_outcome"] = "stuck"
    _budget(budget, d)


def _match_moved(d):
    """One restart's matching loss differs from RQ1's, though its error does not."""
    budget = copy.deepcopy(d["budget"])
    budget[0]["restarts_detail"][3]["match"] *= 2
    _budget(budget, d)


def _batch_two_sizes(d):
    rows = _set(d["batch"], {"n_qubits": 8, "reps": 1, "layers": 4, "best_outcome": "stuck"}, True,
                best_outcome="recovered")  # fmt: skip
    rows = _set(rows, {"n_qubits": 4, "reps": 1, "layers": 2, "batch": 2}, best_outcome="stuck")
    _batch(rows, d)


def _batch_one_circuit(d):
    rows = _set(d["batch"], {"n_qubits": 8, "reps": 1, "layers": 4, "best_outcome": "stuck"}, True,
                best_outcome="recovered")  # fmt: skip
    _batch(rows, d)


def _budget_short(d):
    rows = _set(d["budget"], {"reps": 2, "layers": 1}, restarts=10)
    budget_table(rows, d["tmp"] / "budget.tex", (10, 50))


def _lead_mutant(d):
    """Training leaves every curves D = 2 seed as it was at step 0."""
    rq5 = copy.deepcopy(d["rq5"])
    start = {(*_key(r), r["seed"]): r["best_outcome"] for r in rq5
             if r["task"] == "curves2" and r["step"] == 0}  # fmt: skip
    for r in rq5:
        if r["task"] == "curves2":
            r["best_outcome"] = start[(*_key(r), r["seed"])]
    _trained(rq5, d)


def _majority_unknown_stuck(d):
    """Hiding the label leaves 15 seeds of the 4-qubit 1 x 2, B = 1 circuit stuck."""
    _label(_label_rows(d, (4, 1, 2, 1), "unknown", "recovered", "stuck", 15), d)


def _flip(rows, match, frm, to, n):
    """A copy of rows with the first `n` rows matching `match` and outcome `frm` set to `to`."""
    rows = copy.deepcopy(rows)
    hit = [r for r in rows if r["best_outcome"] == frm and all(r[k] == v for k, v in match.items())]
    assert len(hit) >= n, (match, frm, len(hit))
    for r in hit[:n]:
        r["best_outcome"] = to
    return rows


def _hit_after(rows, circuit, start, n):
    """`n` restarts of `circuit` from index `start` on find the input."""
    rows = copy.deepcopy(rows)
    left = n
    for r in rows:
        if _key(r) == circuit:
            for x in r["restarts_detail"][start:]:
                if left and x["error"] > 0.05:
                    x["error"], left = 0.0, left - 1
    assert left == 0
    return rows


def _square_other_inexact(rows):
    """The restart that found a 1 x 1 input matched worse than the wrong preimage."""
    rows = copy.deepcopy(rows)
    r = next(r for r in rows if _key(r) == (4, 1, 1) and r["any_recovered"]
             and r["best_outcome"] != "recovered")  # fmt: skip
    for d, o in zip(r["restarts_detail"], outcomes(r), strict=True):
        if o == "recovered":
            d["match"] = 1.0
    return rows


def _unstick_deep(rows):
    """At 10^4 shots, 7 of the 8-qubit 4 x 1 circuit's stuck seeds are recovered instead."""
    return _flip(rows, {"n_qubits": 8, "reps": 4, "layers": 1, "shots": 10**4}, "stuck",
                 "recovered", 7)  # fmt: skip


S821 = {"n_qubits": 8, "reps": 2, "layers": 1}
TOP4 = (8, 1, 2, 2)  # the RQ4 setting with most known-only seeds: 8 qubits, 1 x 2, B = 2

# (case, data key, mutation, function, the claim its guard names)
MUTANTS = [
    ("square-exact", "rq1", lambda r: _set(r, {"reps": 1, "layers": 1}, best_match=1.0), _rq1,
     "matched the gradient exactly"),
    ("square-close-large", "rq1", lambda r: _set(r, {"n_qubits": 8, "reps": 1, "layers": 1},
     best_outcome="close", best_error=0.1), _rq1, "no close r = l = 1"),
    ("square-close-small", "rq1", lambda r: _set(r, {"n_qubits": 4, "reps": 1, "layers": 1,
     "best_outcome": "close"}, True, best_outcome="ambiguous"), _rq1, "was close"),
    ("shallow", "rq1", lambda r: _set(r, {"reps": 1, "layers": 4}, best_outcome="stuck"), _rq1,
     "recovered"),
    ("top-line", "rq1", lambda r: _drop(r, {"n_qubits": 4, "reps": 8, "layers": 1}), _rq1,
     "r = 1, 2, 4, 8"),
    ("monotone", "rq1", lambda r: _set(r, {"n_qubits": 8, "reps": 4, "layers": 2}, True,
     best_outcome="recovered"), _rq1, "never lowered"),
    ("rises", "rq1", lambda r: _set(r, {"n_qubits": 4, "reps": 2, "layers": 1,
     "best_outcome": "stuck"}, True, best_outcome="recovered"), _rq1, "largest r was stuck"),
    ("size", "rq1", lambda r: _set(r, {"n_qubits": 8, "reps": 2, "layers": 2,
     "best_outcome": "stuck"}, True, best_outcome="recovered"), _rq1, "more seeds at the larger"),
    ("deep-ambiguous", "rq1", lambda r: _set(r, {"reps": 2, "layers": 2,
     "best_outcome": "recovered"}, best_outcome="ambiguous"), _rq1, "ended ambiguous"),
    ("deep-close", "rq1", lambda r: _set(r, {"reps": 2, "layers": 2,
     "best_outcome": "recovered"}, best_outcome="close"), _rq1, "ended close"),
    ("shot-grid", "rq2", lambda r: _drop(r, {"shots": 100}), _rq2, "100 and 1000"),
    ("low-shot-private", "rq2", lambda r: _set(r, {"n_qubits": 8, "reps": 1, "layers": 4,
     "shots": 100}, True, best_error=0.01), _rq2, "every circuit was private"),
    ("small-leaked", "rq2", lambda r: _set(r, {"n_qubits": 4, "reps": 1, "layers": 4,
     "shots": 10**6}, True, best_error=1.0), _rq2, "at least half"),
    ("private-at-1000", "rq2", lambda r: _set(r, {"n_qubits": 8, "shots": 1000}, True,
     best_error=0.01), _rq2, "not empty"),
    ("wide-full", "rq2", lambda r: _set(_set(_set(r, {"n_qubits": 8, "reps": 1, "layers": 4,
     "shots": 10**4}, best_error=1.0), {"n_qubits": 8, "reps": 1, "layers": 4, "shots": 10**5},
     best_error=1.0), {"n_qubits": 8, "reps": 1, "layers": 4, "shots": 10**6}, best_error=1.0),
     _rq2, "on every seed"),
    ("deep-private", "rq2", lambda r: _set(r, {"n_qubits": 8, "reps": 4, "layers": 1,
     "shots": 10**6}, True, best_error=0.01), _rq2, "at every shot count"),
    ("clean-suffix", "rq2", lambda r: _set(r, {"n_qubits": 8, "reps": 4, "layers": 1,
     "shots": 10**5, "best_outcome": "stuck"}, best_outcome="ambiguous"), _rq2, "no ambiguous"),
    ("same-seeds", "rq2", lambda r: _set(r, {"n_qubits": 4, "reps": 4, "layers": 1,
     "shots": 1000, "best_outcome": "stuck"}, best_outcome="recovered"), _rq2, "same seeds"),
    ("hardest-held", "budget", lambda r: _set(r, {"reps": 8, "layers": 1, "best_outcome":
     "stuck"}, True, best_outcome="recovered"), _budget, "hardest circuit held"),
    ("rose", "budget", lambda r: _set(r, {"reps": 8, "layers": 1, "best_outcome": "recovered"},
     True, best_outcome="stuck"), _budget, "rose with more restarts"),
    ("rate-l1", "budget", lambda r: _hit_after(r, (8, 8, 1), 10, 44), _budget, "along l = 1"),
    ("most-stuck-set", "rq1", lambda r: _set(r, {"n_qubits": 8, "reps": 2, "layers": 2,
     "best_outcome": "recovered"}, True, best_outcome="stuck"), _budget_rq1, "RQ1 circuits"),
    ("only-held", "rq1", lambda r: _set(r, {"n_qubits": 4, "reps": 8, "layers": 1,
     "best_outcome": "recovered"}, True, best_outcome="stuck"), _budget_rq1, "only r = l = 1"),
    ("separate", "batch", lambda r: _set(r, {"n_qubits": 4, "reps": 1, "layers": 2, "batch": 8},
     best_outcome="recovered"), _batch, "does not separate"),
    ("past-crossover", "batch", lambda r: _set(r, {"n_qubits": 4, "reps": 1, "layers": 4,
     "batch": 8}, best_outcome="stuck"), _batch, "every seed was ambiguous"),
    ("below-close", "batch", lambda r: _set(r, {"n_qubits": 8, "reps": 1, "layers": 2,
     "batch": 2, "best_outcome": "stuck"}, best_outcome="close"), _batch, "no failure was close"),
    ("below-ambiguous", "batch", lambda r: _set(r, {"n_qubits": 8, "reps": 1, "layers": 2,
     "batch": 2, "best_outcome": "stuck"}, best_outcome="ambiguous"), _batch,
     "no failure was ambiguous"),
    ("label-ambiguous", "rq4", lambda r: _set(r, {"n_qubits": 8, "reps": 1, "layers": 2,
     "batch": 2, "label": "unknown", "best_outcome": "stuck"}, best_outcome="ambiguous"), _label,
     "were not ambiguous"),
    ("label-close", "rq4", lambda r: _set(r, {"n_qubits": 8, "reps": 1, "layers": 2,
     "batch": 2, "label": "unknown", "best_outcome": "stuck"}, best_outcome="close"), _label,
     "were not close"),
    ("label-significance", "rq4", lambda r: _swap(r, "label", "known", "unknown",
     keep=lambda x: (*_key(x), x["batch"]) == TOP4), _label, "is significant"),
    ("label-direction", "rq4", lambda r: r, lambda m, d: _label(_label_rows(d, (8, 1, 4, 2),
     "known", "recovered", "stuck", 16), d), "recovered fewer inputs"),
    ("label-stuck-equal", "rq4", lambda r: r, lambda m, d: _label(_label_rows(d, TOP4, "unknown",
     "stuck", "ambiguous", 7), d), "more seeds stuck"),
    ("label-one-circuit", "rq4", lambda r: r, lambda m, d: _label(_label_rows(d, (8, 1, 4, 2),
     "unknown", "recovered", "stuck", 6), d), "mostly on one circuit"),
    ("trained-outcomes", "rq5", lambda r: _set(r, {"step": 300}, best_outcome="close"), _trained,
     "recovered or stuck"),
    ("trained-shallow", "rq5", lambda r: _flip(r, {"task": "linear", "n_qubits": 4, "reps": 1,
     "step": 10}, "recovered", "stuck", 5), _trained, "fewest 1 x l"),
    ("trained-shallow-most", "rq5", lambda r: _flip(r, {"task": "linear", "n_qubits": 4,
     "reps": 1, "step": 10}, "recovered", "stuck", 11), _trained, "on most seeds"),
    ("trained-shallow-layers", "rq5", lambda r: [{**x, "layers": 2} if x["reps"] == 1 and
     x["task"] == "linear" and x["n_qubits"] == 4 else x for x in r], _trained, "is 1 x 4"),
    ("trained-shallow-tie", "rq5", lambda r: _flip(r, {"task": "linear", "n_qubits": 4,
     "reps": 1, "step": 10}, "recovered", "stuck", 4), _trained, "one setting holds"),
    ("square-both-exact", "rq1", _square_other_inexact, _rq1, "also matched exactly"),
    ("noise-signal", "rq2", lambda r: _set(r, {"n_qubits": 8, "shots": 100}, True,
     true_match=0.0), _rq2, "as large as the signal"),
    ("low-shot-ambiguous", "rq2", lambda r: _flip(r, {"n_qubits": 8, "reps": 1, "layers": 4,
     "shots": 100}, "ambiguous", "stuck", 15), _rq2, "were ambiguous"),
    ("still-stuck", "rq2", _unstick_deep, _rq2, "moved few"),
    ("half", "rq1", lambda r: _flip(r, {"n_qubits": 4, "reps": 8, "layers": 1}, "recovered",
     "stuck", 2), _budget_rq1, "exactly half"),
    ("bracket", "batch", lambda r: _set(_set(r, {"n_qubits": 4, "reps": 1, "layers": 4,
     "batch": 8}, best_outcome="recovered"), {"n_qubits": 8, "reps": 1, "layers": 4, "batch": 8},
     best_outcome="recovered"), _batch, "brackets one unknown"),
    ("batch-all-recovered", "batch", lambda r: _set(r, {"n_qubits": 8, "reps": 1, "layers": 4,
     "batch": 4}, True, best_outcome="recovered"), _batch, "left more seeds stuck"),
]  # fmt: skip

WHOLE = [
    ("rate-120", _rate_120, "largest parameter count"),
    ("repeat-error", _first_seed_not_best, r"\(error\)"),
    ("repeat-count", _rq1_outcome_flipped, "reproduce the RQ1 count"),
    ("most-stuck-recovered", _stuck_not_recovered, "were recovered with more restarts"),
    ("repeat-match", _match_moved, r"\(match\)"),
    ("lead", _lead_mutant, "smallest per-task p"),
    ("lead-direction", lambda d: _trained_swap(d, ("curves2",)), "fewer curves2"),
    ("lead-others", _linear_significant, "other per-task tests"),
    ("trained-overall", lambda d: _trained_swap(d, ("linear", "curves10")), "no overall change"),
    ("trained-gradient", _shrunk_gradient, "shrink the median gradient"),
    ("label-batch", _majority_unknown_stuck, "largest batch"),
    ("label-significant", _significant_setting, "significant"),
    ("no-slope", _no_slope, "came within CLOSE_RAD at three"),
    ("batch-stuck-falls", _batch_stuck_falls, "left more seeds stuck"),
    ("batch-two-sizes", _batch_two_sizes, "at one size"),
    ("batch-one-circuit", _batch_one_circuit, "on two circuits"),
]  # fmt: skip


@pytest.fixture
def data(real, tmp_path):
    return {**real, "tmp": tmp_path}


def test_real_results_pass_every_guard(data):
    for key, fn in (("rq1", _rq1), ("rq2", _rq2), ("budget", _budget), ("batch", _batch),
                    ("rq4", _label), ("rq5", _trained)):  # fmt: skip
        fn(data[key], data)


@pytest.mark.parametrize(
    ("case", "key", "mutate", "fn", "claim"), MUTANTS, ids=[m[0] for m in MUTANTS]
)
def test_each_guard_fires_on_its_mutant(data, case, key, mutate, fn, claim):
    with pytest.raises(macros.ClaimError, match=claim):
        fn(mutate(data[key]), data)


@pytest.mark.parametrize(("case", "run", "claim"), WHOLE, ids=[m[0] for m in WHOLE])
def test_each_cross_file_guard_fires_on_its_mutant(data, case, run, claim):
    with pytest.raises(macros.ClaimError, match=claim):
        run(data)


def test_budget_table_refuses_runs_shorter_than_the_budget(data):
    with pytest.raises(ValueError, match="fewer restarts"):
        _budget_short(data)
