"""Feasibility spike: DLG-style gradient inversion on a tiny PennyLane VQC."""
import time
import pennylane as qml
from pennylane import numpy as pnp
import numpy as np
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
            for l in range(layers):
                for q in range(N):
                    qml.RY(th[r, l, q, 0], wires=q)
                    qml.RZ(th[r, l, q, 1], wires=q)
                for q in range(N - 1):
                    qml.CNOT(wires=[q, q + 1])
        return qml.expval(qml.PauliZ(0))

    def g(x, th, y):  # dL/dθ, L = (f - y)^2
        return qml.grad(lambda t: (circ(x, t) - y) ** 2, argnums=0)(th)

    return g


def attack(reps, layers, restarts, seed):
    rng = np.random.default_rng(seed)
    g = make(reps, layers)
    P = reps * layers * N * 2
    th = pnp.array(rng.uniform(0, 2 * np.pi, P), requires_grad=True)
    xs = rng.uniform(0, np.pi, N)
    y = 1.0
    g_obs = np.array(g(pnp.array(xs, requires_grad=False), th, y))

    def loss(xv):
        d = g(xv, th, y) - g_obs
        return pnp.sum(d**2)

    dloss = qml.grad(loss, argnums=0)
    best = None
    for _ in range(restarts):
        r = minimize(
            lambda v: float(loss(pnp.array(v, requires_grad=True))),
            rng.uniform(0, np.pi, N),
            jac=lambda v: np.array(dloss(pnp.array(v, requires_grad=True))),
            method="L-BFGS-B",
            options={"maxiter": 200},
        )
        if best is None or r.fun < best.fun:
            best = r
    return P, best.fun, np.linalg.norm(np.cos(best.x) - np.cos(xs))


for reps, layers in [(1, 1), (1, 4), (4, 1), (4, 4)]:
    t = time.time()
    res = [attack(reps, layers, restarts=5, seed=s) for s in range(5)]
    print(
        f"reps={reps} layers={layers} params={res[0][0]:3d}  "
        f"recon_err={np.round([r[2] for r in res], 3)}  "
        f"match={[f'{r[1]:.0e}' for r in res]}  {time.time() - t:.0f}s",
        flush=True,
    )
