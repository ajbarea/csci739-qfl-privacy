<div align="center">

# csci739-qfl-privacy

*When do variational quantum circuits keep a federated client's training data private?*

[![CI](https://github.com/ajbarea/csci739-qfl-privacy/actions/workflows/ci.yml/badge.svg)](https://github.com/ajbarea/csci739-qfl-privacy/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.12%20%7C%203.13%20%7C%203.14-blue)](pyproject.toml)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

</div>

---

In federated learning, clients share gradients instead of data, and on classical networks those
gradients can be inverted to rebuild the training inputs. For variational quantum circuits the
literature disagrees: [Kumar et al. (2023)](https://arxiv.org/abs/2309.13002) argue expressive
encodings make inversion intractable, while
[Papadopoulos et al. (2025)](https://arxiv.org/abs/2504.12806) reconstruct inputs from
overparameterized circuits. This project maps where each claim holds, across encoding depth,
trainable layers, batch size, and simulated IBM Heron device noise.

Final project for CSCI-739 Quantum Machine Learning, RIT, Fall 2026.

## First result

A 4-qubit spike (one sample, known label, untrained weights, 5 seeds) already separates three
outcomes: the input recovered, a *different* input found with the identical gradient, or the attack
stuck in a local minimum.

| Encoding reps x trainable layers | Parameters | Recovered | Same gradient, wrong input | Stuck |
|---|---|---|---|---|
| 1 x 1 | 8 | 1 of 5, plus 1 within 0.07 | 2 | 1 |
| 1 x 4 | 32 | 5 of 5 | 0 | 0 |
| 4 x 1 | 32 | 3 of 5 | 0 | 2 |
| 4 x 4 | 128 | 4 of 5 | 0 | 1 |

At equal parameter count, the more expressive encoding (4 x 1) resists the attack more often than
the deeper trainable block (1 x 4). Raw output:
[`spike/results-2026-09-24.txt`](spike/results-2026-09-24.txt).

## Run it

```bash
uv sync --all-groups
uv run python spike/gradient_inversion.py   # the full spike
uv run pytest -q                            # fast regression of both regimes
```

The attack uses PennyLane, which differentiates through the simulator (the attack needs the
gradient of a gradient). Device noise comes from Qiskit Aer with the offline calibration snapshots
of `ibm_kingston`, `ibm_fez`, and `ibm_marrakesh`, so no IBM Quantum account is required.
