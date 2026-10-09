"""Gradient-matching inversion (Zhu et al. 2019) against the VQC client update."""

from dataclasses import asdict, dataclass
from functools import cache

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import linear_sum_assignment, minimize

from qflp.circuit import Circuit

# A restart lands on the true input when every feature is within this angle of it, mod 2π.
RECOVERED_RAD = 0.05
# Under shot noise the exact input is out of reach; a reconstruction this close still leaks it.
CLOSE_RAD = 0.2
OUTCOMES = ("recovered", "close", "ambiguous", "stuck")
# A wrong input "explains" the gradient when it matches it at least this well relative to the truth.
EXPLAINS_SLACK = 1e-6
# Tikhonov term, relative to the mean squared norm of the per-sample gradients, that keeps the
# label-free solve defined when two of them are parallel. Far below EXPLAINS_SLACK.
RIDGE = 1e-12


def circular_error(x_hat, x_true) -> float:
    """Largest per-feature distance mod 2π, minimized over orderings of the batch.

    RY(x + 2π) = -RY(x) is a global phase, so inputs are only defined mod 2π. Batch order is not
    observable from a mean gradient, so rows are matched up to permutation: the bottleneck
    assignment, found by bisecting on the threshold with a perfect-matching check.
    """
    x_hat, x_true = np.atleast_2d(x_hat), np.atleast_2d(x_true)
    if not np.isfinite(x_hat).all():
        return float("inf")
    diff = np.angle(np.exp(1j * (x_hat[:, None, :] - x_true[None, :, :])))
    dist = np.abs(diff).max(-1)  # dist[i, j]: row i of x_hat against row j of x_true
    levels = np.unique(dist)
    lo, hi = 0, len(levels) - 1
    while lo < hi:
        mid = (lo + hi) // 2
        rows, cols = linear_sum_assignment((dist > levels[mid]).astype(float))
        if (dist[rows, cols] > levels[mid]).any():
            lo = mid + 1
        else:
            hi = mid
    return float(levels[lo])


@cache
def _loss_and_grad(circuit: Circuit, label_known: bool = True):
    """Jitted value-and-gradient of the matching loss; compiled once per circuit shape.

    With the label known, the loss is the squared distance to the client's gradient. Without it,
    each sample's weight c_i = 2 (f_i - y_i) / B is free, so the loss is the squared distance from
    the client's gradient to the span of the candidate inputs' output gradients (the best c in
    closed form). For B = 1 this is Geiping et al.'s scale-invariant matching.
    """

    def loss(x_flat, theta, g_obs, ys):
        xs = x_flat.reshape(ys.shape[0], circuit.n_qubits)
        if label_known:
            d = circuit.batch_grad(xs, theta, ys) - g_obs
        else:
            jac = circuit.output_grads(xs, theta)
            gram = jac @ jac.T
            ridge = RIDGE * jnp.trace(gram) / ys.shape[0] * jnp.eye(ys.shape[0])
            c = jnp.linalg.solve(gram + ridge, jac @ g_obs)
            d = jac.T @ c - g_obs
        return (d**2).sum()

    return jax.jit(jax.value_and_grad(loss))


def matching_loss(circuit: Circuit, xs, theta, g_obs, ys, label_known: bool = True) -> float:
    vg = _loss_and_grad(circuit, label_known)
    return float(vg(jnp.ravel(jnp.asarray(xs)), theta, g_obs, ys)[0])


@dataclass
class Restart:
    match: float
    error: float
    nit: int
    outcome: str


def classify(error: float, match: float, true_match: float, scale: float) -> str:
    """recovered: the true input; close: within CLOSE_RAD of it, which is what a noisy gradient
    allows; ambiguous: a different input that fits the gradient at least as well as the true one;
    stuck: a local minimum that fits worse."""
    if error <= RECOVERED_RAD:
        return "recovered"
    if error <= CLOSE_RAD:
        return "close"
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
    label_known: bool = True,
) -> tuple[list[Restart], float]:
    """Run `restarts` L-BFGS-B searches from uniform[0, π) starts.

    Returns every restart and the matching loss of the true input, which is 0 for an exact gradient
    and positive once the client's gradient is noisy. Without `label_known`, `ys` only sets the
    batch size; the attack never reads the labels.
    """
    theta, g_obs, ys = jnp.asarray(theta), jnp.asarray(g_obs), jnp.asarray(ys, dtype=float)
    vg = _loss_and_grad(circuit, label_known)
    x_true = np.atleast_2d(x_true)
    true_match = matching_loss(circuit, x_true, theta, g_obs, ys, label_known)
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
    return min(restarts, key=lambda r: r.match if np.isfinite(r.match) else np.inf)


def as_dict(r: Restart) -> dict:
    return asdict(r)
