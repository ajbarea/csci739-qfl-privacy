"""Run the attack over a grid of circuits, seeds, and shot budgets; append one JSON line per run.

    uv run python -m qflp.sweep --qubits 4 8 --grid 1x1 1x2 2x1 --seeds 20 --out results/rq1.jsonl

Resumable: rows already in --out (same qubits, reps, layers, seed, shots, batch, restarts) are
skipped, so an interrupted sweep continues where it stopped.
"""

import argparse
import json
import multiprocessing as mp
import os
import platform
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

KEY = ("n_qubits", "reps", "layers", "seed", "shots", "batch", "restarts")


def run_one(job: dict) -> dict:
    import jax.numpy as jnp
    import numpy as np

    from qflp.attack import as_dict, attack, best_restart
    from qflp.circuit import Circuit
    from qflp.noise import shot_gradient

    c = Circuit(job["n_qubits"], job["reps"], job["layers"])
    # One generator per run, seeded from the run's identity, so any row can be re-run alone.
    rng = np.random.default_rng(
        [job["seed"], job["n_qubits"], job["reps"], job["layers"], job["batch"]]
    )
    theta = rng.uniform(0, 2 * np.pi, c.n_params)
    xs = rng.uniform(0, np.pi, (job["batch"], c.n_qubits))
    ys = np.ones(job["batch"])
    if job["shots"]:
        shot_rng = np.random.default_rng([job["seed"], job["shots"]])
        g_obs = np.mean(
            [shot_gradient(c, x, theta, 1.0, job["shots"], shot_rng) for x in xs], axis=0
        )
    else:
        g_obs = np.asarray(c.batch_grad(jnp.asarray(xs), jnp.asarray(theta), jnp.asarray(ys)))
    t0 = time.perf_counter()
    restarts, true_match = attack(c, theta, g_obs, ys, xs, job["restarts"], rng)
    best = best_restart(restarts)
    return {
        **job,
        "n_params": c.n_params,
        "n_effective": c.n_effective,
        "true_match": true_match,
        "grad_norm2": float((g_obs**2).sum()),
        "best_outcome": best.outcome,
        "best_match": best.match,
        "best_error": best.error,
        "any_recovered": any(r.outcome == "recovered" for r in restarts),
        "restarts_detail": [as_dict(r) for r in restarts],
        "seconds": time.perf_counter() - t0,
    }


def _done(path: Path) -> set[tuple]:
    if not path.exists():
        return set()
    with path.open() as fh:
        return {tuple(row[k] for k in KEY) for row in map(json.loads, fh)}


def _commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--qubits", type=int, nargs="+", default=[4])
    ap.add_argument("--grid", nargs="+", default=["1x4", "4x1"], help="REPSxLAYERS")
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--restarts", type=int, default=10)
    ap.add_argument("--shots", type=int, nargs="+", default=[0], help="0 = exact gradient")
    ap.add_argument("--batch", type=int, nargs="+", default=[1])
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)

    jobs = [
        {
            "n_qubits": n,
            "reps": int(g.split("x")[0]),
            "layers": int(g.split("x")[1]),
            "seed": s,
            "shots": shots,
            "batch": b,
            "restarts": args.restarts,
        }
        for n in args.qubits
        for g in args.grid
        for shots in args.shots
        for b in args.batch
        for s in range(args.seeds)
    ]
    done = _done(args.out)
    todo = [j for j in jobs if tuple(j[k] for k in KEY) not in done]
    print(f"{len(jobs)} runs, {len(jobs) - len(todo)} already in {args.out}, {len(todo)} to go")  # noqa: T201
    args.out.parent.mkdir(parents=True, exist_ok=True)
    meta = {"commit": _commit(), "python": platform.python_version()}
    # spawn, not fork: JAX is multithreaded and fork after import can deadlock.
    ctx = mp.get_context("spawn")
    with (
        ProcessPoolExecutor(args.workers, mp_context=ctx) as pool,
        args.out.open("a") as fh,
    ):
        futures = [pool.submit(run_one, j) for j in todo]
        for i, fut in enumerate(as_completed(futures), 1):
            row = {**fut.result(), **meta}
            fh.write(json.dumps(row) + "\n")
            fh.flush()
            print(  # noqa: T201
                f"[{i}/{len(todo)}] {row['n_qubits']}q {row['reps']}x{row['layers']} "
                f"seed={row['seed']} shots={row['shots']} batch={row['batch']}: "
                f"{row['best_outcome']} ({row['seconds']:.1f}s)",
                flush=True,
            )


if __name__ == "__main__":
    main()
