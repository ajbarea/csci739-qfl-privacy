"""Gradient-matching inversion (Zhu et al. 2019) against the VQC client update."""

import itertools
from dataclasses import asdict, dataclass
from functools import cache

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize

from qflp.circuit import Circuit

# A restart lands on the true input when every feature is within this angle of it, mod 2π.
RECOVERED_RAD = 0.05
# A wrong input "explains" the gradient when it matches it at least this well relative to the truth.
EXPLAINS_SLACK = 1e-6


def circular_error(x_hat, x_true) -> float:
    """Largest per-feature distance mod 2π, minimized over orderings of the batch.

    RY(x + 2π) = -RY(x) is a global phase, so inputs are only defined mod 2π. Batch order is not
    observable from a mean gradient, so rows are matched up to permutation.
    """
    x_hat, x_true = np.atleast_2d(x_hat), np.atleast_2d(x_true)
    best = np.inf
    for perm in itertools.permutations(range(len(x_true))):
        d = np.angle(np.exp(1j * (x_hat[list(perm)] - x_true)))
        best = min(best, float(np.abs(d).max()))
    return best


@cache
def _loss_and_grad(circuit: Circuit):
    """Jitted value-and-gradient of the matching loss; compiled once per circuit shape."""

    def loss(x_flat, theta, g_obs, ys):
        xs = x_flat.reshape(ys.shape[0], circuit.n_qubits)
        d = circuit.batch_grad(xs, theta, ys) - g_obs
        return (d**2).sum()

    return jax.jit(jax.value_and_grad(loss))


def matching_loss(circuit: Circuit, xs, theta, g_obs, ys) -> float:
    vg = _loss_and_grad(circuit)
    return float(vg(jnp.ravel(jnp.asarray(xs)), theta, g_obs, ys)[0])


@dataclass
class Restart:
    match: float
    error: float
    nit: int
    outcome: str


def classify(error: float, match: float, true_match: float, scale: float) -> str:
    """recovered: the true input; ambiguous: a wrong input that fits the gradient at least as well
    as the true one; stuck: a local minimum that fits worse."""
    if error <= RECOVERED_RAD:
        return "recovered"
    if match <= true_match + EXPLAINS_SLACK * scale:
        return "ambiguous"
    return "stuck"


def attack(
    circuit: Circuit,
    theta,
    g_obs,
    ys,
    x_true,
    restarts: int,
    rng: np.random.Generator,
    maxiter: int = 500,
) -> tuple[list[Restart], float]:
    """Run `restarts` L-BFGS-B searches from uniform[0, π) starts.

    Returns every restart and the matching loss of the true input, which is 0 for an exact gradient
    and positive once the client's gradient is noisy.
    """
    theta, g_obs, ys = jnp.asarray(theta), jnp.asarray(g_obs), jnp.asarray(ys, dtype=float)
    vg = _loss_and_grad(circuit)
    x_true = np.atleast_2d(x_true)
    true_match = matching_loss(circuit, x_true, theta, g_obs, ys)
    scale = float((g_obs**2).sum())

    def fun(v):
        val, grad = vg(jnp.asarray(v), theta, g_obs, ys)
        return float(val), np.asarray(grad, dtype=float)

    out = []
    for _ in range(restarts):
        res = minimize(
            fun,
            rng.uniform(0, np.pi, x_true.size),
            jac=True,
            method="L-BFGS-B",
            options={"maxiter": maxiter, "ftol": 1e-15, "gtol": 1e-12},
        )
        err = circular_error(res.x.reshape(x_true.shape), x_true)
        out.append(
            Restart(
                match=float(res.fun),
                error=err,
                nit=int(res.nit),
                outcome=classify(err, float(res.fun), true_match, scale),
            )
        )
    return out, true_match


def best_restart(restarts: list[Restart]) -> Restart:
    """What an attacker without the ground truth would keep: the lowest matching loss."""
    return min(restarts, key=lambda r: r.match)


def as_dict(r: Restart) -> dict:
    return asdict(r)
