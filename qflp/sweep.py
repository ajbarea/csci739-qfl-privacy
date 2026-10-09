"""Run the attack over a grid of circuits, seeds, and shot budgets; append one JSON line per run.

    uv run python -m qflp.sweep --qubits 4 8 --grid 1x1 1x2 2x1 --seeds 20 --out results/rq1.jsonl

POSIX only: the output file is locked with flock while a sweep runs.

Resumable: rows already in --out (same qubits, reps, layers, seed, shots, batch, restarts, label)
are skipped, so an interrupted sweep continues where it stopped.

Label modes: `fixed`, every y = 1 and the attacker knows it (the first sweeps); `known`, each y
drawn from {-1, +1} and given to the attacker; `unknown`, the same draw, hidden from the attacker.
Runs that differ only in label mode share θ, inputs and the attack's starting points.
"""

import argparse
import fcntl
import json
import multiprocessing as mp
import os
import platform
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from qflp.rows import KEY, read_rows


def run_one(job: dict) -> dict:
    import jax.numpy as jnp
    import numpy as np

    from qflp.attack import as_dict, attack, best_restart
    from qflp.circuit import Circuit
    from qflp.noise import shot_gradient

    c = Circuit(job["n_qubits"], job["reps"], job["layers"])
    # One generator per run, seeded from the run's identity, so any row can be re-run alone.
    identity = [job["seed"], job["n_qubits"], job["reps"], job["layers"], job["batch"]]
    rng = np.random.default_rng(identity)
    theta = rng.uniform(0, 2 * np.pi, c.n_params)
    xs = rng.uniform(0, np.pi, (job["batch"], c.n_qubits))
    if job["label"] == "fixed":
        ys = np.ones(job["batch"])
    else:
        # A child stream, so `known` and `unknown` see the same labels and `fixed` is unchanged.
        label_rng = np.random.default_rng(np.random.SeedSequence(identity).spawn(1)[0])
        ys = label_rng.choice([-1.0, 1.0], job["batch"])
    label_known = job["label"] != "unknown"
    if job["shots"]:
        # Separate stream so the attack's starting points match the exact-gradient run.
        shot_rng = np.random.default_rng([*identity, job["shots"]])
        g_obs = np.mean(
            [
                shot_gradient(c, x, theta, y, job["shots"], shot_rng)
                for x, y in zip(xs, ys, strict=True)
            ],
            axis=0,
        )
    else:
        g_obs = np.asarray(c.batch_grad(jnp.asarray(xs), jnp.asarray(theta), jnp.asarray(ys)))
    t0 = time.perf_counter()
    # Without the label the attacker gets placeholder labels, which the label-free loss ignores.
    ys_attacker = ys if label_known else np.ones(job["batch"])
    restarts, true_match = attack(
        c, theta, g_obs, ys_attacker, xs, job["restarts"], rng, label_known=label_known
    )
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
    """Keys of the rows already in `path`, after dropping a partial last line."""
    if not path.exists():
        return set()
    return {tuple(row[k] for k in KEY) for row in read_rows(path, repair=True)}


def _commit() -> str:
    """Short hash of the checkout this package runs from, +dirty if qflp/ has local changes."""
    here = Path(__file__).resolve().parent
    try:

        def git(*cmd: str) -> str:
            return subprocess.run(
                ["git", "-C", str(here), *cmd], capture_output=True, text=True, check=True
            ).stdout.strip()

        dirty = git("status", "--porcelain", "--untracked-files=no", "--", ".")
        return git("rev-parse", "--short", "HEAD") + ("+dirty" if dirty else "")
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
    ap.add_argument("--label", nargs="+", default=["fixed"], choices=["fixed", "known", "unknown"])
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    if args.out.suffix != ".jsonl":
        ap.error(f"--out must be a .jsonl results file, got {args.out}")

    jobs = [
        {
            "n_qubits": n,
            "reps": int(g.split("x")[0]),
            "layers": int(g.split("x")[1]),
            "seed": s,
            "shots": shots,
            "batch": b,
            "restarts": args.restarts,
            "label": label,
        }
        for n in args.qubits
        for g in args.grid
        for shots in args.shots
        for b in args.batch
        for label in args.label
        for s in range(args.seeds)
    ]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    meta = {"commit": _commit(), "python": platform.python_version()}
    # spawn, not fork: JAX is multithreaded and fork after import can deadlock.
    ctx = mp.get_context("spawn")
    with args.out.open("a") as fh:
        # Held from the resume repair to the last append, so two sweeps never share one file.
        try:
            fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            ap.error(f"another sweep is writing {args.out}")
        done = _done(args.out)
        todo = [j for j in jobs if tuple(j[k] for k in KEY) not in done]
        print(f"{len(jobs)} runs, {len(jobs) - len(todo)} already in {args.out}, {len(todo)} to go")  # noqa: T201
        with ProcessPoolExecutor(args.workers, mp_context=ctx) as pool:
            futures = {pool.submit(run_one, j): j for j in todo}
            failed = 0
            for i, fut in enumerate(as_completed(futures), 1):
                try:
                    row = {**fut.result(), **meta}
                except Exception as exc:  # one bad run must not discard the rest
                    failed += 1
                    print(f"[{i}/{len(todo)}] FAILED {futures[fut]}: {exc!r}", flush=True)  # noqa: T201
                    continue
                fh.write(json.dumps(row) + "\n")
                fh.flush()
                print(  # noqa: T201
                    f"[{i}/{len(todo)}] {row['n_qubits']}q {row['reps']}x{row['layers']} "
                    f"seed={row['seed']} shots={row['shots']} batch={row['batch']}: "
                    f"{row['best_outcome']} ({row['seconds']:.1f}s)",
                    flush=True,
                )
    if failed:
        raise SystemExit(f"{failed} of {len(todo)} runs failed; rerun to retry them")


if __name__ == "__main__":
    main()
