"""Time the first (compiling) and a warm call of the attack's loss-and-gradient for one shape.

uv run python -m qflp.compile_time --qubits 8 --grid 4x4 8x1 1x8
"""

import argparse
import time

import jax
import jax.numpy as jnp
import numpy as np

from qflp.attack import _loss_and_grad
from qflp.circuit import Circuit


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--qubits", type=int, default=8)
    ap.add_argument("--grid", nargs="+", default=["4x4", "8x1", "1x8"], help="REPSxLAYERS")
    args = ap.parse_args(argv)
    rng = np.random.default_rng(0)
    for g in args.grid:
        reps, layers = map(int, g.split("x"))
        c = Circuit(args.qubits, reps, layers)
        vg = _loss_and_grad(c)
        theta = jnp.asarray(rng.uniform(0, 2 * np.pi, c.n_params))
        ys = jnp.ones(1)
        x = jnp.asarray(rng.uniform(0, np.pi, c.n_qubits))
        g_obs = c.batch_grad(x[None], theta, ys)
        times = []
        for v in (x, x + 0.1):  # first call compiles; the second reuses the compiled function
            t0 = time.perf_counter()
            jax.block_until_ready(vg(v, theta, g_obs, ys))
            times.append(time.perf_counter() - t0)
        print(f"{args.qubits}q {g}: first call {times[0]:.2f} s, warm call {times[1] * 1e3:.1f} ms")  # noqa: T201


if __name__ == "__main__":
    main()
