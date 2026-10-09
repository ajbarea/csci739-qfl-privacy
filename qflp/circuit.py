"""Statevector simulator for the re-uploading VQC, written in JAX so the attack can jit.

The circuit matches `spike/gradient_inversion.py` and the PennyLane reference in the tests:
`reps` times, encode every feature as RY(x_q) on qubit q, then apply `layers` trainable blocks of
RY(θ) RZ(θ) on every qubit followed by a CNOT chain 0→1→…→N-1. The output is the mean of Z over
all qubits. Wire 0 is the most significant bit, as in PennyLane.
"""

from dataclasses import dataclass
from functools import cached_property

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)


def _apply_1q(psi, mats, n):
    """Apply one 2x2 matrix per qubit; `mats` has shape (n, 2, 2)."""
    for q in range(n):
        psi = jnp.moveaxis(jnp.tensordot(mats[q], psi, axes=([1], [q])), 0, q)
    return psi


def _ry(angles):
    c, s = jnp.cos(angles / 2), jnp.sin(angles / 2)
    return jnp.stack([jnp.stack([c, -s], -1), jnp.stack([s, c], -1)], -2).astype(jnp.complex128)


def _rz(angles):
    e = jnp.exp(-0.5j * angles)
    z = jnp.zeros_like(e)
    return jnp.stack([jnp.stack([e, z], -1), jnp.stack([z, jnp.conj(e)], -1)], -2)


@dataclass(frozen=True)
class Circuit:
    n_qubits: int
    reps: int
    layers: int

    @property
    def n_params(self) -> int:
        return self.reps * self.layers * self.n_qubits * 2

    @property
    def n_effective(self) -> int:
        """Parameters with a nonzero gradient.

        The last block's RZ gates are followed only by CNOTs and a Z-diagonal measurement, so they
        commute with the observable and never change the output.
        """
        return self.n_params - self.n_qubits

    @cached_property
    def _chain_perm(self) -> np.ndarray:
        """Basis-state permutation of the whole CNOT chain, as a gather index."""
        n = self.n_qubits
        idx = np.arange(2**n)
        bits = (idx[:, None] >> (n - 1 - np.arange(n))) & 1
        for q in range(n - 1):
            bits[:, q + 1] ^= bits[:, q]
        out = (bits << (n - 1 - np.arange(n))).sum(1)
        perm = np.empty_like(out)
        perm[out] = idx  # new[out[i]] = old[i]  →  new = old[perm]
        return perm

    @cached_property
    def z_mean(self) -> np.ndarray:
        n = self.n_qubits
        idx = np.arange(2**n)
        bits = (idx[:, None] >> (n - 1 - np.arange(n))) & 1
        return (1 - 2 * bits).mean(1).astype(float)

    def probs(self, x, theta):
        """Computational-basis probabilities after the circuit."""
        n, perm = self.n_qubits, jnp.asarray(self._chain_perm)
        th = theta.reshape(self.reps, self.layers, n, 2)
        enc = _ry(x)

        def block(psi, t):
            psi = _apply_1q(psi, _ry(t[:, 0]), n)
            psi = _apply_1q(psi, _rz(t[:, 1]), n)
            return psi.reshape(-1)[perm].reshape((2,) * n), None

        def rep(psi, t_rep):
            psi = _apply_1q(psi, enc, n)
            psi, _ = jax.lax.scan(block, psi, t_rep)
            return psi, None

        psi0 = jnp.zeros((2,) * n, jnp.complex128).at[(0,) * n].set(1.0)
        psi, _ = jax.lax.scan(rep, psi0, th)
        return jnp.abs(psi.reshape(-1)) ** 2

    def expval(self, x, theta):
        return self.probs(x, theta) @ jnp.asarray(self.z_mean)

    def grad(self, x, theta, y):
        """Client update: dL/dθ for L = (f(x; θ) - y)^2."""
        return jax.grad(lambda t: (self.expval(x, t) - y) ** 2)(theta)

    def batch_grad(self, xs, theta, ys):
        """Mean per-sample gradient over a batch (rows of `xs`)."""
        return jax.vmap(self.grad, in_axes=(0, None, 0))(xs, theta, ys).mean(0)

    def output_grads(self, xs, theta):
        """df/dθ for each row of `xs`, shape (batch, n_params). The client's batch gradient is
        sum_i c_i * row i, with c_i = 2 (f(x_i) - y_i) / batch."""
        return jax.vmap(jax.grad(self.expval, argnums=1), in_axes=(0, None))(xs, theta)
