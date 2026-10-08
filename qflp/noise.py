"""The client's gradient as a device would estimate it: parameter shift with finite shots."""

import jax
import jax.numpy as jnp
import numpy as np

from qflp.circuit import Circuit


def shot_gradient(
    circuit: Circuit, x, theta, y: float, shots: int, rng: np.random.Generator
) -> np.ndarray:
    """dL/dθ for L = (f - y)^2, with f and every ∂f/∂θ_j estimated from `shots` measurements.

    Every gate is RY or RZ, so ∂f/∂θ_j = [f(θ + π/2 e_j) - f(θ - π/2 e_j)] / 2 exactly. Each of the
    2P + 1 expectation values is an independent `shots`-sample estimate of mean Z, drawn from the
    exact basis-state probabilities.
    """
    theta = jnp.asarray(theta)
    p = circuit.n_params
    shifts = jnp.concatenate([jnp.zeros((1, p)), jnp.eye(p) * np.pi / 2, -jnp.eye(p) * np.pi / 2])
    probs = np.asarray(jax.vmap(lambda s: circuit.probs(jnp.asarray(x), theta + s))(shifts))
    probs = np.clip(probs, 0, None)
    probs /= probs.sum(1, keepdims=True)
    z = circuit.z_mean
    est = np.array([rng.multinomial(shots, pr) @ z / shots for pr in probs])
    f, plus, minus = est[0], est[1 : p + 1], est[p + 1 :]
    return 2 * (f - y) * (plus - minus) / 2
