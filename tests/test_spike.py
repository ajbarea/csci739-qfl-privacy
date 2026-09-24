import numpy as np
import pytest
from pennylane import numpy as pnp

from spike.gradient_inversion import N, attack, make


@pytest.mark.parametrize(("reps", "layers"), [(1, 1), (1, 4), (4, 1)])
def test_every_input_reaches_the_gradient(reps, layers):
    rng = np.random.default_rng(0)
    g = make(reps, layers)
    th = pnp.array(rng.uniform(0, 2 * np.pi, reps * layers * N * 2), requires_grad=True)
    x = rng.uniform(0, np.pi, N)
    base = np.array(g(pnp.array(x, requires_grad=False), th, 1.0))
    for i in range(N):
        moved = x.copy()
        moved[i] += 0.7
        shifted = np.array(g(pnp.array(moved, requires_grad=False), th, 1.0))
        assert np.abs(shifted - base).max() > 1e-6, f"x[{i}] never reaches the gradient"


def test_small_circuit_gradient_has_a_second_exact_preimage():
    _, match, err = attack(reps=1, layers=1, restarts=5, seed=0)
    assert match < 1e-10
    assert err > 1.0


def test_deep_trainable_circuit_input_is_recovered():
    _, match, err = attack(reps=1, layers=4, restarts=5, seed=0)
    assert match < 1e-10
    assert err < 1e-3
