"""Train the client's circuit on a benchmark task, and pick its learning rate on a validation split.

    uv run python -m qflp.train --out results/rq5-training.jsonl

Writes one row per (task, qubits, reps x layers, learning rate, tuning seed) with train, validation
and test accuracy, and one baseline row per (task, qubits) for logistic regression. The sweep reads
the learning rate with the best mean validation accuracy for each circuit from this file
(`qflp.rows.best_lr`).
"""

import argparse
import json
from functools import cache
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax

from qflp.circuit import Circuit
from qflp.data import TASKS, make_task

STEPS = 300  # full-batch Adam steps; the loss curves in the tuning rows show the plateau
LEARNING_RATES = (0.001, 0.01, 0.1)  # the grid Bowles et al. search for re-uploading models
TUNING_SEEDS = 3
VALIDATION = 0.2  # fraction of the training split held out for tuning


@cache
def _step(circuit: Circuit, lr: float):
    opt = optax.adam(lr)

    def loss(theta, xs, ys):
        f = jax.vmap(circuit.expval, in_axes=(0, None))(xs, theta)
        return ((f - ys) ** 2).mean()

    @jax.jit
    def step(theta, state, xs, ys):
        value, grad = jax.value_and_grad(loss)(theta, xs, ys)
        updates, state = opt.update(grad, state, theta)
        return optax.apply_updates(theta, updates), state, value

    return opt, step


def train(
    circuit: Circuit, theta0, xs, ys, lr: float, steps: int
) -> tuple[np.ndarray, list[float]]:
    """θ after `steps` full-batch Adam steps on mean (f - y)^2, and the loss before each step."""
    opt, step = _step(circuit, lr)
    theta = jnp.asarray(theta0)
    state = opt.init(theta)
    xs, ys = jnp.asarray(xs), jnp.asarray(ys)
    losses = []
    for _ in range(steps):
        theta, state, value = step(theta, state, xs, ys)
        losses.append(float(value))
    return np.asarray(theta), losses


def accuracy(circuit: Circuit, theta, xs, ys) -> float:
    f = jax.vmap(circuit.expval, in_axes=(0, None))(jnp.asarray(xs), jnp.asarray(theta))
    return float((np.sign(np.asarray(f)) == ys).mean())


def initial_theta(circuit: Circuit, seed: int) -> np.ndarray:
    """θ drawn as in RQ1, uniform on [0, 2π), from the seed alone."""
    return np.random.default_rng([seed, circuit.n_qubits, circuit.reps, circuit.layers]).uniform(
        0, 2 * np.pi, circuit.n_params
    )


def tuning_rows(task: str, circuit: Circuit, data_seed: int):
    x_tr, x_te, y_tr, y_te = make_task(task, circuit.n_qubits, data_seed)
    n_val = round(VALIDATION * len(y_tr))
    x_fit, y_fit, x_val, y_val = x_tr[n_val:], y_tr[n_val:], x_tr[:n_val], y_tr[:n_val]
    for lr in LEARNING_RATES:
        for seed in range(TUNING_SEEDS):
            theta, losses = train(
                circuit, initial_theta(circuit, 1000 + seed), x_fit, y_fit, lr, STEPS
            )
            yield {
                "task": task,
                "n_qubits": circuit.n_qubits,
                "reps": circuit.reps,
                "layers": circuit.layers,
                "lr": lr,
                "tuning_seed": seed,
                "train_acc": accuracy(circuit, theta, x_fit, y_fit),
                "val_acc": accuracy(circuit, theta, x_val, y_val),
                "test_acc": accuracy(circuit, theta, x_te, y_te),
                "losses": [*losses[:: max(1, STEPS // 30)], losses[-1]],
            }


def baseline_row(task: str, n_qubits: int, data_seed: int) -> dict:
    from sklearn.linear_model import LogisticRegression

    x_tr, x_te, y_tr, y_te = make_task(task, n_qubits, data_seed)
    model = LogisticRegression(max_iter=10_000).fit(x_tr, y_tr)
    return {
        "task": task,
        "n_qubits": n_qubits,
        "model": "logistic",
        "train_acc": float(model.score(x_tr, y_tr)),
        "test_acc": float(model.score(x_te, y_te)),
    }


DATA_SEED = 0  # one dataset per (task, qubits), as in the benchmark


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--qubits", type=int, nargs="+", default=[4, 8])
    ap.add_argument("--grid", nargs="+", default=["1x4", "4x1"], help="REPSxLAYERS")
    ap.add_argument("--tasks", nargs="+", default=list(TASKS), choices=list(TASKS))
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as fh:
        for task in args.tasks:
            for n in args.qubits:
                fh.write(json.dumps(baseline_row(task, n, DATA_SEED)) + "\n")
                for g in args.grid:
                    reps, layers = map(int, g.split("x"))
                    for row in tuning_rows(task, Circuit(n, reps, layers), DATA_SEED):
                        fh.write(json.dumps(row) + "\n")
                        fh.flush()
                        print(  # noqa: T201
                            f"{task} {n}q {g} lr={row['lr']} seed={row['tuning_seed']}: "
                            f"train {row['train_acc']:.2f} val {row['val_acc']:.2f} "
                            f"loss {row['losses'][-1]:.3f}",
                            flush=True,
                        )


if __name__ == "__main__":
    main()
