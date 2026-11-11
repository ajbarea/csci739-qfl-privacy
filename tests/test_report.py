"""The report's generated tables and prose numbers match results/, and every claim guard fires."""

import copy
import json
import re
from pathlib import Path

import pytest

from qflp import macros
from qflp.figures import batch_table, label_table, trained_table
from qflp.figures import main as figures_main
from qflp.summarize import best_outcome, load

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


def _rate_mutant(d):
    """Every restart of the 8-qubit 4 x 2 circuit misses, in RQ1 and in the budget rerun alike."""
    rq1, budget = copy.deepcopy(d["rq1"]), copy.deepcopy(d["budget"])
    for rows in (rq1, budget):
        for r in rows:
            if (r["n_qubits"], r["reps"], r["layers"]) == (8, 4, 2):
                for x in r["restarts_detail"]:
                    x["error"] = 3.0
        _rescore([r for r in rows if (r["n_qubits"], r["reps"], r["layers"]) == (8, 4, 2)])
    macros.budget_macros(budget, rq1)


def _lead_mutant(d):
    """Training leaves every curves D = 2 seed as it was at step 0."""
    rq5 = copy.deepcopy(d["rq5"])
    start = {(r["n_qubits"], r["reps"], r["layers"], r["seed"]): r["best_outcome"] for r in rq5
             if r["task"] == "curves2" and r["step"] == 0}  # fmt: skip
    for r in rq5:
        if r["task"] == "curves2":
            r["best_outcome"] = start[(r["n_qubits"], r["reps"], r["layers"], r["seed"])]
    _trained(rq5, d)


def _majority_unknown_stuck(d):
    """Hiding the label leaves 15 seeds of the 4-qubit 1 x 2, B = 1 circuit stuck."""
    rq4 = copy.deepcopy(d["rq4"])
    hit = [r for r in rq4 if (r["n_qubits"], r["reps"], r["layers"], r["batch"], r["label"])
           == (4, 1, 2, 1, "unknown") and r["best_outcome"] == "recovered"]  # fmt: skip
    for r in hit[:15]:
        r["best_outcome"] = "stuck"
    _label(rq4, d)


def _significant_setting(d):
    """Every 8-qubit 1 x 2, B = 2 seed is recovered with the label and stuck without it."""
    rq4 = _set(d["rq4"], {"n_qubits": 8, "reps": 1, "layers": 2, "batch": 2, "label": "known"},
               True, best_outcome="recovered")  # fmt: skip
    rq4 = _set(rq4, {"n_qubits": 8, "reps": 1, "layers": 2, "batch": 2, "label": "unknown"},
               True, best_outcome="stuck")  # fmt: skip
    _label(rq4, d)


def _match_differs(d):
    budget = copy.deepcopy(d["budget"])
    budget[0]["restarts_detail"][3]["match"] *= 2
    _budget(budget, d)


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
     "best_outcome": "stuck"}, True, best_outcome="recovered"), _rq1, "never lowered"),
    ("size", "rq1", lambda r: _set(r, {"n_qubits": 8, "reps": 2, "layers": 2,
     "best_outcome": "stuck"}, True, best_outcome="recovered"), _rq1, "more seeds"),
    ("deep-ambiguous", "rq1", lambda r: _set(r, {"reps": 2, "layers": 2,
     "best_outcome": "recovered"}, best_outcome="ambiguous"), _rq1, "every failure was stuck"),
    ("deep-close", "rq1", lambda r: _set(r, {"reps": 2, "layers": 2,
     "best_outcome": "recovered"}, best_outcome="close"), _rq1, "every failure was stuck"),
    ("shot-grid", "rq2", lambda r: _drop(r, {"shots": 100}), _rq2, "100 and 1000"),
    ("low-shot-private", "rq2", lambda r: _set(r, {"n_qubits": 8, "reps": 1, "layers": 4,
     "shots": 100}, True, best_error=0.01), _rq2, "every circuit was private"),
    ("small-leaked", "rq2", lambda r: _set(r, {"n_qubits": 4, "reps": 1, "layers": 4,
     "shots": 10**6}, True, best_error=1.0), _rq2, "at least half"),
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
    ("repeat-error", "budget", lambda r: _set(r, {"reps": 2, "layers": 1, "seed": 0},
     restarts_detail=[{**x, "error": x["error"] + 0.5} for x in r[0]["restarts_detail"]]),
     _budget, "repeat the RQ1 attack"),
    ("hardest-held", "budget", lambda r: _set(r, {"reps": 8, "layers": 1, "best_outcome":
     "stuck"}, True, best_outcome="recovered"), _budget, "hardest circuit held"),
    ("most-stuck", "rq1", lambda r: _set(r, {"n_qubits": 8, "reps": 2, "layers": 2,
     "best_outcome": "recovered"}, True, best_outcome="stuck"), _budget_rq1, "most stuck"),
    ("only-held", "rq1", lambda r: _set(r, {"n_qubits": 4, "reps": 8, "layers": 1,
     "best_outcome": "recovered"}, True, best_outcome="stuck"), _budget_rq1, "only r = l = 1"),
    ("past-crossover", "batch", lambda r: _set(r, {"n_qubits": 4, "reps": 1, "layers": 4,
     "batch": 8}, best_outcome="stuck"), _batch, "every seed was ambiguous"),
    ("below-crossover", "batch", lambda r: _set(r, {"n_qubits": 8, "reps": 1, "layers": 2,
     "batch": 2, "best_outcome": "stuck"}, best_outcome="close"), _batch, "every failure"),
    ("stuck-circuits", "batch", lambda r: _set(r, {"n_qubits": 4, "reps": 1, "layers": 2,
     "batch": 2}, best_outcome="stuck"), _batch, "two circuits of one size"),
    ("label-stuck", "rq4", lambda r: _set(r, {"n_qubits": 8, "reps": 1, "layers": 2, "batch": 2,
     "label": "unknown", "best_outcome": "stuck"}, best_outcome="ambiguous"), _label,
     "stuck, not ambiguous"),
    ("trained-outcomes", "rq5", lambda r: _set(r, {"step": 300}, best_outcome="close"), _trained,
     "recovered or stuck"),
    ("trained-shallow", "rq5", lambda r: _set(r, {"task": "linear", "n_qubits": 4, "reps": 1,
     "step": 10}, True, best_outcome="stuck"), _trained, "fewest 1 x l"),
]  # fmt: skip

WHOLE = [
    ("rate", _rate_mutant, "lower per-restart rate"),
    ("rate-l1", lambda d: _budget(_rescore([{**r, "restarts_detail": [
        {**x, "error": 0.0} if i >= 10 else x for i, x in enumerate(r["restarts_detail"])]}
        if (r["reps"], r["layers"]) == (8, 1) else r for r in copy.deepcopy(d["budget"])]), d),
     "lower per-restart rate"),
    ("lead", _lead_mutant, "smallest per-task p"),
    ("label-batch", _majority_unknown_stuck, "largest batch"),
    ("label-significant", _significant_setting, "significant"),
    ("repeat-match", _match_differs, "repeat the RQ1 attack"),
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
