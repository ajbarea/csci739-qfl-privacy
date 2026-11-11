"""The report's generated tables and prose numbers match results/, and every claim guard fires."""

import copy
import re
from pathlib import Path

import pytest

from qflp import macros
from qflp.figures import main as figures_main
from qflp.summarize import load

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
    source = (REPORT / "report.tex").read_text()
    used = set(re.findall(r"\\([A-Z]\w*)(?![A-Za-z])", source))
    assert defined - used == set(), "generated but never quoted"
    # Capitalised commands the report uses that are neither generated nor LaTeX's own.
    latex = {"IEEEauthorblockN", "IEEEauthorblockA", "IEEEkeywords"}
    assert used - defined - latex == set(), "quoted but not generated"


@pytest.mark.parametrize(
    ("x", "text"),
    [
        (1.93e-6, "\\ensuremath{2\\times10^{-6}}"),
        (1e-6, "\\ensuremath{10^{-6}}"),
        (9.5e-3, "\\ensuremath{10^{-2}}"),
        (3.0, "\\ensuremath{3\\times10^{0}}"),
    ],
)
def test_upper_sci_rounds_up_to_one_figure(x, text):
    assert macros._upper_sci(x) == text


def test_pow10_refuses_a_value_that_is_not_a_power_of_ten():
    assert macros._pow10(1e-3) == "\\ensuremath{10^{-3}}"
    with pytest.raises(ValueError, match="power of ten"):
        macros._pow10(2e-3)


@pytest.fixture(scope="module")
def rows():
    return {k: load([RESULTS / f"{k}.jsonl"]) for k in ("rq1", "rq2", "budget")}


def _set(rows, match, every=False, **fields):
    """A copy of rows with `fields` set on the first row matching `match`, or on every one."""
    rows = copy.deepcopy(rows)
    hits = [r for r in rows if all(r[k] == v for k, v in match.items())]
    assert hits
    for r in hits if every else hits[:1]:
        r.update(fields)
    return rows


# Each mutant breaks one sentence of the report; its guard must refuse to write the macros.
def test_rq1_guards_fire(rows):
    rq1 = rows["rq1"]
    macros.rq1_macros(rq1)
    for mutant, claim in [
        (_set(rq1, {"reps": 2, "layers": 1}, best_outcome="ambiguous"), "ended ambiguous"),
        (_set(rq1, {"reps": 1, "layers": 4}, best_outcome="stuck"), "recovered"),
        (_set(rq1, {"reps": 1, "layers": 1}, best_match=1.0), "matched the gradient exactly"),
        (
            _set(rq1, {"n_qubits": 8, "reps": 4, "layers": 2}, True, best_outcome="recovered"),
            "never lowered",
        ),
        (
            _set(rq1, {"n_qubits": 8, "reps": 2, "layers": 2, "best_outcome": "stuck"},
                 every=True, best_outcome="recovered"),
            "more seeds",
        ),
    ]:  # fmt: skip
        with pytest.raises(macros.ClaimError, match=claim):
            macros.rq1_macros(mutant)


def test_rq2_guards_fire(rows):
    rq1, rq2 = rows["rq1"], rows["rq2"]
    macros.rq2_macros(rq2, rq1)
    shifted = _set(rq2, {"n_qubits": 4, "reps": 4, "layers": 1, "shots": 1000,
                         "best_outcome": "stuck"}, best_outcome="recovered")  # fmt: skip
    with pytest.raises(macros.ClaimError, match="same seeds"):
        macros.rq2_macros(shifted, rq1)
    leaky = _set(rq2, {"n_qubits": 8, "reps": 4, "layers": 1, "shots": 10**6}, every=True,
                 best_error=0.01)  # fmt: skip
    with pytest.raises(macros.ClaimError, match="private on most seeds at every shot count"):
        macros.rq2_macros(leaky, rq1)


def test_budget_guard_fires_when_the_first_restarts_differ_from_rq1(rows):
    budget, rq1 = rows["budget"], rows["rq1"]
    macros.budget_macros(budget, rq1)
    mutant = copy.deepcopy(budget)
    mutant[0]["restarts_detail"][3]["error"] += 0.5
    with pytest.raises(macros.ClaimError, match="repeat the RQ1 attack"):
        macros.budget_macros(mutant, rq1)


def test_budget_guard_fires_when_another_circuit_holds_most_seeds(rows):
    rq1 = _set(rows["rq1"], {"n_qubits": 4, "reps": 8, "layers": 1, "best_outcome": "recovered"},
               every=True, best_outcome="stuck")  # fmt: skip
    with pytest.raises(macros.ClaimError, match="only r = l = 1"):
        macros.budget_macros(rows["budget"], rq1)


def test_batch_guard_fires_when_a_seed_past_the_crossover_is_not_ambiguous(rows, tmp_path):
    from qflp.figures import batch_table

    rq3 = load([RESULTS / "rq3.jsonl"])
    circuits = {(r["n_qubits"], r["reps"], r["layers"]) for r in rq3}
    batch = [r for r in rows["rq1"] if (r["n_qubits"], r["reps"], r["layers"]) in circuits] + rq3
    batch_table(batch, tmp_path / "rq3.tex")
    mutant = _set(batch, {"n_qubits": 4, "reps": 1, "layers": 4, "batch": 8}, best_outcome="stuck")
    with pytest.raises(macros.ClaimError, match="every seed was ambiguous"):
        batch_table(mutant, tmp_path / "rq3.tex")


def test_budget_guard_fires_when_more_repetitions_raise_the_per_restart_rate(rows):
    mutant = copy.deepcopy(rows["budget"])
    for r in mutant:
        if (r["reps"], r["layers"]) == (8, 1):
            for d in r["restarts_detail"][10:]:
                d["error"] = 0.0
    with pytest.raises(macros.ClaimError, match="lower per-restart rate"):
        macros.budget_macros(mutant, rows["rq1"])
