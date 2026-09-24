"""Feasibility spike: DLG-style gradient inversion on a tiny PennyLane VQC."""

import time

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp
from scipy.optimize import minimize

N = 4
dev = qml.device("default.qubit", wires=N)


def make(reps, layers):
    @qml.qnode(dev, diff_method="backprop")
    def circ(x, th):
        th = th.reshape(reps, layers, N, 2)
        for r in range(reps):  # data re-uploading = encoding expressivity
            for q in range(N):
                qml.RY(x[q], wires=q)
            for layer in range(layers):
                for q in range(N):
                    qml.RY(th[r, layer, q, 0], wires=q)
                    qml.RZ(th[r, layer, q, 1], wires=q)
                for q in range(N - 1):
                    qml.CNOT(wires=[q, q + 1])
        return qml.expval(qml.PauliZ(0))

    def g(x, th, y):  # dL/dθ, L = (f - y)^2
        return qml.grad(lambda t: (circ(x, t) - y) ** 2, argnums=0)(th)

    return g


def attack(reps, layers, restarts, seed):
    """Return (param count, final gradient-match loss, input reconstruction error)."""
    rng = np.random.default_rng(seed)
    g = make(reps, layers)
    P = reps * layers * N * 2
    th = pnp.array(rng.uniform(0, 2 * np.pi, P), requires_grad=True)
    xs = rng.uniform(0, np.pi, N)
    y = 1.0
    g_obs = np.array(g(pnp.array(xs, requires_grad=False), th, y))

    def loss(xv):
        d = g(xv, th, y) - g_obs
        return (d**2).sum()

    dloss = qml.grad(loss, argnums=0)
    runs = [
        minimize(
            lambda v: float(loss(pnp.array(v, requires_grad=True))),
            rng.uniform(0, np.pi, N),
            jac=lambda v: np.array(dloss(pnp.array(v, requires_grad=True))),
            method="L-BFGS-B",
            options={"maxiter": 200},
        )
        for _ in range(restarts)
    ]
    best = min(runs, key=lambda r: r.fun)
    # RY encoding cannot distinguish x from 2π - x, so compare cos x.
    return P, best.fun, np.linalg.norm(np.cos(best.x) - np.cos(xs))


if __name__ == "__main__":
    for reps, layers in [(1, 1), (1, 4), (4, 1), (4, 4)]:
        t = time.time()
        res = [attack(reps, layers, restarts=5, seed=s) for s in range(5)]
        print(  # noqa: T201
            f"reps={reps} layers={layers} params={res[0][0]:3d}  "
            f"recon_err={np.round([r[2] for r in res], 3)}  "
            f"match={[f'{r[1]:.0e}' for r in res]}  {time.time() - t:.0f}s",
            flush=True,
        )
