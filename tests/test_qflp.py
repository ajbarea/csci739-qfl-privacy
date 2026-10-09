import itertools
import json

import jax.numpy as jnp
import numpy as np
import pennylane as qml
import pytest

from qflp.attack import CLOSE_RAD, RECOVERED_RAD, attack, best_restart, circular_error, classify
from qflp.circuit import Circuit
from qflp.noise import shot_gradient
from qflp.summarize import read_rows, table
from qflp.sweep import _done


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
    row = {"n_qubits": 4, "reps": 1, "layers": 1, "seed": 0, "shots": 0, "batch": 1, "restarts": 3}
    path = tmp_path / "out.jsonl"
    path.write_text(json.dumps(row) + "\n" + json.dumps({**row, "seed": 1})[:20])
    assert _done(path) == {tuple(row.values())}
    assert path.read_text() == json.dumps(row) + "\n"
    assert len(read_rows(path)) == 1


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
    rows = [
        {**base, "seed": 0, "best_outcome": "recovered", "any_recovered": True, "best_error": 0.0},
        {**base, "seed": 1, "best_outcome": "stuck", "any_recovered": True, "best_error": 2.0},
        {**base, "seed": 2, "best_outcome": "ambiguous", "any_recovered": False, "best_error": 0.1},
    ]
    (s,) = table([{**r, "n_effective": 4} for r in rows])
    assert (s["recovered"], s["ambiguous"], s["stuck"], s["any_restart_recovered"]) == (1, 1, 1, 2)
    assert (s["median_error"], s["within_close"]) == (0.1, 2)
