import fcntl
import itertools
import json

import jax.numpy as jnp
import numpy as np
import pennylane as qml
import pytest

from qflp.attack import (
    CLOSE_RAD,
    RECOVERED_RAD,
    attack,
    best_restart,
    circular_error,
    classify,
    matching_loss,
)
from qflp.circuit import Circuit
from qflp.noise import shot_gradient
from qflp.rows import DEFAULTS, KEY, ROW_START, read_rows
from qflp.summarize import load, table
from qflp.sweep import _done, run_one
from qflp.sweep import main as sweep_main


def pennylane_reference(c: Circuit):
    dev = qml.device("default.qubit", wires=c.n_qubits)

    @qml.qnode(dev, diff_method="backprop")
    def probs(x, th):
        th = th.reshape(c.reps, c.layers, c.n_qubits, 2)
        for r in range(c.reps):
            for q in range(c.n_qubits):
                qml.RY(x[q], wires=q)
            for layer in range(c.layers):
                for q in range(c.n_qubits):
                    qml.RY(th[r, layer, q, 0], wires=q)
                    qml.RZ(th[r, layer, q, 1], wires=q)
                for q in range(c.n_qubits - 1):
                    qml.CNOT(wires=[q, q + 1])
        return qml.probs(wires=range(c.n_qubits))

    return probs


def random_point(c: Circuit, seed: int):
    rng = np.random.default_rng(seed)
    return rng.uniform(0, np.pi, c.n_qubits), rng.uniform(0, 2 * np.pi, c.n_params)


@pytest.mark.parametrize(("n", "reps", "layers"), [(2, 1, 1), (3, 2, 3), (4, 3, 2), (5, 1, 2)])
def test_simulator_matches_pennylane(n, reps, layers):
    c = Circuit(n, reps, layers)
    x, th = random_point(c, n + reps + layers)
    ref = np.asarray(pennylane_reference(c)(x, th))
    assert np.abs(ref - np.asarray(c.probs(jnp.asarray(x), jnp.asarray(th)))).max() < 1e-12


def test_gradient_matches_pennylane():
    c = Circuit(4, 2, 2)
    x, th = random_point(c, 7)
    circuit = pennylane_reference(c)
    z = qml.numpy.array(c.z_mean, requires_grad=False)

    def loss(t):
        return ((circuit(x, t) * z).sum() - 1.0) ** 2

    ref = np.asarray(qml.grad(loss)(qml.numpy.array(th, requires_grad=True)))
    ours = np.asarray(c.grad(jnp.asarray(x), jnp.asarray(th), 1.0))
    assert np.abs(ref - ours).max() < 1e-12


@pytest.mark.parametrize(("n", "reps", "layers"), [(4, 1, 1), (4, 4, 1), (8, 1, 2)])
def test_effective_params_are_all_but_the_last_rz(n, reps, layers):
    c = Circuit(n, reps, layers)
    nonzero = np.zeros(c.n_params, dtype=bool)
    for seed in range(3):
        x, th = random_point(c, seed)
        nonzero |= np.abs(np.asarray(c.grad(jnp.asarray(x), jnp.asarray(th), 1.0))) > 1e-10
    assert nonzero.sum() == c.n_effective
    last_rz = np.zeros((reps, layers, n, 2), dtype=bool)
    last_rz[-1, -1, :, 1] = True
    assert not nonzero[last_rz.ravel()].any()


@pytest.mark.parametrize(("n", "reps", "layers"), [(4, 1, 1), (4, 1, 4), (4, 4, 1), (8, 1, 1)])
def test_every_input_reaches_the_gradient(n, reps, layers):
    c = Circuit(n, reps, layers)
    x, th = random_point(c, 0)
    base = np.asarray(c.grad(jnp.asarray(x), jnp.asarray(th), 1.0))
    for i in range(n):
        moved = x.copy()
        moved[i] += 0.7
        shifted = np.asarray(c.grad(jnp.asarray(moved), jnp.asarray(th), 1.0))
        assert np.abs(shifted - base).max() > 1e-6, f"x[{i}] never reaches the gradient"


def test_circular_error_is_mod_2pi_and_order_free():
    x = np.array([[0.1, 0.2], [1.0, 2.0]])
    assert circular_error(x + 2 * np.pi, x) < 1e-12
    assert circular_error(x[::-1], x) < 1e-12
    y = np.array([1.0, 0.5])
    assert circular_error(-y, y) == pytest.approx(2.0)  # cos would fold -y onto y


def test_circular_error_matches_brute_force_over_permutations():
    rng = np.random.default_rng(0)
    for batch in (1, 2, 3, 5):
        for _ in range(20):
            x = rng.uniform(0, 2 * np.pi, (batch, 3))
            x_hat = x[rng.permutation(batch)] + rng.normal(0, 0.3, (batch, 3))
            brute = min(
                float(np.abs(np.angle(np.exp(1j * (x_hat[list(p)] - x)))).max())
                for p in itertools.permutations(range(batch))
            )
            assert circular_error(x_hat, x) == pytest.approx(brute)


def test_classify():
    assert classify(RECOVERED_RAD / 2, 1.0, 0.0, 1.0) == "recovered"
    assert classify((RECOVERED_RAD + CLOSE_RAD) / 2, 1.0, 0.0, 1.0) == "close"
    assert classify(1.0, 1e-12, 0.0, 1.0) == "ambiguous"
    assert classify(1.0, 1e-3, 0.0, 1.0) == "stuck"
    # Under noise the truth no longer matches exactly; a far input that fits as well is ambiguous,
    # a near one that fits better is a close reconstruction, not a second preimage.
    assert classify(1.0, 0.5, 0.5, 1.0) == "ambiguous"
    assert classify(0.1, 0.4, 0.5, 1.0) == "close"


def test_resume_drops_a_partial_last_line(tmp_path):
    row = {"n_qubits": 4, "reps": 1, "layers": 1, "shots": 0, "batch": 1, "restarts": 3, "seed": 0}
    path = tmp_path / "out.jsonl"
    path.write_text(json.dumps(row) + "\n" + json.dumps({**row, "seed": 1})[:20])
    assert _done(path) == {tuple({**DEFAULTS, **row}[k] for k in KEY)}
    assert path.read_text() == json.dumps(row) + "\n"
    assert len(read_rows(path)) == 1


def test_resume_refuses_a_file_that_is_not_sweep_rows(tmp_path):
    path = tmp_path / "notes.jsonl"
    text = "\\section{Results}\nsome prose\n"
    path.write_text(text)
    with pytest.raises(ValueError, match="not a sweep row"):
        _done(path)
    assert path.read_text() == text  # never truncated


def test_sweep_refuses_a_non_jsonl_output(tmp_path):
    target = tmp_path / "report.tex"
    target.write_text("prose")
    with pytest.raises(SystemExit):
        sweep_main(["--out", str(target), "--seeds", "0"])
    assert target.read_text() == "prose"


def test_resume_terminates_a_complete_last_row_before_appending(tmp_path):
    row = {"n_qubits": 4, "reps": 1, "layers": 1, "shots": 0, "batch": 1, "restarts": 3, "seed": 0}
    path = tmp_path / "out.jsonl"
    path.write_text(json.dumps(row))  # complete row, no final newline
    assert _done(path) == {tuple({**DEFAULTS, **row}[k] for k in KEY)}
    assert path.read_text() == json.dumps(row) + "\n"


def test_repair_keeps_the_file_identity(tmp_path):
    row = {"n_qubits": 4, "reps": 1, "layers": 1, "shots": 0, "batch": 1, "restarts": 3, "seed": 0}
    target = tmp_path / "real.jsonl"
    target.write_text(json.dumps(row) + "\n" + '{"n_qubits": 4, "re')
    link = tmp_path / "link.jsonl"
    link.symlink_to(target)
    inode = target.stat().st_ino
    _done(link)
    assert link.is_symlink()
    assert target.stat().st_ino == inode
    assert target.read_text() == json.dumps(row) + "\n"


@pytest.mark.parametrize("sep", ["\n", "\r", "\r\n"])
def test_repair_truncates_only_the_cut_off_row(tmp_path, sep):
    row = {"n_qubits": 4, "reps": 1, "layers": 1, "shots": 0, "batch": 1, "restarts": 3, "seed": 0}
    kept = json.dumps(row) + sep + json.dumps({**row, "seed": 1}) + sep
    path = tmp_path / "out.jsonl"
    path.write_bytes((kept + '{"n_qubits": 4, "re').encode())
    assert len(_done(path)) == 2
    assert path.read_bytes() == kept.encode()


@pytest.mark.parametrize("text", ["{\\bf hello}", '{"a": 1}', '{"a": 1}\n', '{"a":1} trailing'])
def test_repair_leaves_non_rows_untouched(tmp_path, text):
    path = tmp_path / "notes.jsonl"
    path.write_text(text)
    with pytest.raises(ValueError, match="not a sweep row"):
        _done(path)
    assert path.read_text() == text


def test_a_sweep_row_starts_as_repair_expects(tmp_path):
    path = tmp_path / "out.jsonl"
    args = ["--qubits", "2", "--grid", "1x1", "--seeds", "1", "--restarts", "1", "--workers", "1"]
    sweep_main([*args, "--out", str(path)])
    (line,) = path.read_text().splitlines()
    assert line.startswith(ROW_START)
    assert all(k in json.loads(line) for k in KEY)


def test_sweep_refuses_a_file_another_sweep_holds(tmp_path):
    path = tmp_path / "out.jsonl"
    path.write_text('{"n_qubits": 4, "re')
    with path.open("a") as other:
        fcntl.flock(other, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(SystemExit):
            sweep_main(["--out", str(path), "--seeds", "0"])
    assert path.read_text() == '{"n_qubits": 4, "re'  # not repaired under someone else's lock


def test_a_lone_non_json_line_is_not_emptied(tmp_path):
    path = tmp_path / "notes.jsonl"
    path.write_text("TODO: notes")
    with pytest.raises(ValueError, match="not a sweep row"):
        _done(path)
    assert path.read_text() == "TODO: notes"


def test_load_refuses_duplicate_runs(tmp_path):
    row = {
        "n_qubits": 4,
        "reps": 1,
        "layers": 1,
        "shots": 0,
        "batch": 1,
        "restarts": 1,
        "seed": 0,
        "true_match": 0.0,
        "grad_norm2": 1.0,
        "restarts_detail": [{"error": 0.0, "match": 0.0, "nit": 1, "outcome": "recovered"}],
    }
    path = tmp_path / "a.jsonl"
    path.write_text(json.dumps(row) + "\n")
    with pytest.raises(ValueError, match="duplicate"):
        load([path, path])


def test_circular_error_of_a_non_finite_guess_is_infinite():
    x = np.zeros((2, 3))
    assert circular_error(np.full((2, 3), np.nan), x) == np.inf


def test_deep_trainable_circuit_input_is_recovered():
    c = Circuit(4, 1, 4)
    x, th = random_point(c, 0)
    g = c.grad(jnp.asarray(x), jnp.asarray(th), 1.0)
    restarts, true_match = attack(c, th, g, [1.0], x, 5, np.random.default_rng(0))
    assert true_match == 0.0
    best = best_restart(restarts)
    assert best.outcome == "recovered"
    assert best.match < 1e-15


def test_shot_gradient_converges_to_exact():
    c = Circuit(4, 1, 2)
    x, th = random_point(c, 3)
    exact = np.asarray(c.grad(jnp.asarray(x), jnp.asarray(th), 1.0))
    errs = [
        np.abs(shot_gradient(c, x, th, 1.0, s, np.random.default_rng(0)) - exact).max()
        for s in (10**2, 10**4, 10**6)
    ]
    assert errs[0] > errs[1] > errs[2]
    assert errs[2] < 1e-2


def test_summary_counts_best_restart_outcomes():
    base = {"n_qubits": 4, "reps": 1, "layers": 1, "shots": 0, "batch": 1, "restarts": 5}
    base["label"] = "fixed"
    rows = [
        {**base, "seed": 0, "best_outcome": "recovered", "any_recovered": True, "best_error": 0.0},
        {**base, "seed": 1, "best_outcome": "stuck", "any_recovered": True, "best_error": 2.0},
        {**base, "seed": 2, "best_outcome": "ambiguous", "any_recovered": False, "best_error": 0.1},
    ]
    (s,) = table([{**r, "n_effective": 4} for r in rows])
    assert (s["recovered"], s["ambiguous"], s["stuck"], s["any_restart_recovered"]) == (1, 1, 1, 2)
    assert (s["median_error"], s["within_close"]) == (0.1, 2)


def test_rows_from_before_the_label_axis_load_as_fixed(tmp_path):
    row = {"n_qubits": 4, "reps": 1, "layers": 1, "shots": 0, "batch": 1, "restarts": 3, "seed": 0}
    path = tmp_path / "old.jsonl"
    path.write_text(json.dumps(row) + "\n")
    (loaded,) = read_rows(path)
    assert loaded["label"] == "fixed"


@pytest.mark.parametrize("batch", [1, 2, 3])
def test_label_free_loss_is_zero_at_the_truth_and_never_above_the_known_label_loss(batch):
    c = Circuit(3, 1, 2)
    rng = np.random.default_rng(batch)
    theta = jnp.asarray(rng.uniform(0, 2 * np.pi, c.n_params))
    xs = jnp.asarray(rng.uniform(0, np.pi, (batch, 3)))
    ys = jnp.asarray(rng.choice([-1.0, 1.0], batch))
    g = c.batch_grad(xs, theta, ys)
    scale = float((g**2).sum())
    placeholder = jnp.ones(batch)
    assert matching_loss(c, xs, theta, g, placeholder, label_known=False) <= 1e-12 * scale
    for _ in range(5):
        guess = jnp.asarray(rng.uniform(0, np.pi, (batch, 3)))
        free = matching_loss(c, guess, theta, g, placeholder, label_known=False)
        known = matching_loss(c, guess, theta, g, ys, label_known=True)
        assert free <= known * (1 + 1e-9) + 1e-15


def test_label_modes_share_the_client_and_hide_the_label():
    job = {"n_qubits": 2, "reps": 1, "layers": 2, "seed": 3, "shots": 0, "batch": 2, "restarts": 1}
    rows = {m: run_one({**job, "label": m}) for m in ("fixed", "known", "unknown")}
    assert rows["known"]["grad_norm2"] == rows["unknown"]["grad_norm2"]
    assert rows["fixed"]["grad_norm2"] != rows["known"]["grad_norm2"]
    assert rows["unknown"]["true_match"] <= 1e-12 * rows["unknown"]["grad_norm2"]


def test_mcnemar_exact():
    from qflp.figures import mcnemar_exact

    assert mcnemar_exact(0, 0) == 1.0
    assert mcnemar_exact(9, 2) == mcnemar_exact(2, 9) == pytest.approx(0.0654, abs=1e-4)
    assert mcnemar_exact(10, 0) == pytest.approx(2 / 2**10)


def test_label_table_pools_discordant_seeds_per_batch(tmp_path):
    from qflp.figures import label_table

    def row(n, b, label, seed, ok):
        return {
            "n_qubits": n, "reps": 1, "layers": 2, "shots": 0, "batch": b, "restarts": 10,
            "label": label, "seed": seed, "n_effective": 12, "best_outcome": "recovered" if ok
            else "stuck", "any_recovered": ok, "best_error": 0.0 if ok else 2.0,
        }  # fmt: skip

    # Circuit 4q: seed 0 recovered only with the label. Circuit 8q: seed 0 only with, seed 1 only
    # without. Batch 1 agrees everywhere.
    rows = [
        row(4, 2, "known", 0, True), row(4, 2, "unknown", 0, False),
        row(8, 2, "known", 0, True), row(8, 2, "unknown", 0, False),
        row(8, 2, "known", 1, False), row(8, 2, "unknown", 1, True),
        row(4, 1, "known", 0, True), row(4, 1, "unknown", 0, True),
    ]  # fmt: skip
    macros = label_table(rows, tmp_path / "rq4.tex")
    assert (macros["LabelKnownOnlyTwo"], macros["LabelUnknownOnlyTwo"]) == ("2", "1")
    assert (macros["LabelKnownOnlyOne"], macros["LabelUnknownOnlyOne"]) == ("0", "0")
    assert macros["LabelPooledPTwo"] == "1.000"
    assert (macros["LabelKnownOnlyAll"], macros["LabelUnknownOnlyAll"]) == ("2", "1")
    assert macros["LabelKnownOnlyTopTwo"] == "1"
    assert len((tmp_path / "rq4.tex").read_text().splitlines()) == 3
