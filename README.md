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

## Results so far

Gradient-matching attacks on 4- and 8-qubit re-uploading circuits (an attacker who knows the
circuit, its parameters and the label; one sample; 20 seeds per setting; raw rows in
[`results/`](results), write-up in [`report/report.tex`](report/report.tex)):

- Every circuit with one encoding layer and two or more trainable layers gave up its input on 20 of
  20 seeds.
- At a fixed parameter count, encoding repetitions leave more attacks stuck. At 8 qubits and 120
  effective parameters, 20, 18, 8 and 2 of 20 seeds were recovered for 1, 2, 4 and 8 repetitions.
- With 50 restarts instead of 10, most stuck seeds are recovered (8 qubits: 7→18, 8→17, 8→18,
  2→5), so encoding repetitions raise the attacker's cost rather than stopping the attack.
- Shot noise protects 8-qubit circuits at about 100 shots per expectation value. For circuits the
  noiseless attack breaks, the reconstruction error then falls as about S^-1/2.

## Run it

```bash
uv sync --all-groups
uv run pytest -q
uv run python -m qflp.sweep --qubits 4 --grid 1x4 4x1 --seeds 3 --out /tmp/qflp-demo.jsonl
uv run python -m qflp.summarize /tmp/qflp-demo.jsonl
```

The attack runs on `qflp`, a small JAX statevector simulator, because it needs the gradient of a
gradient and must compile once per circuit shape. Tests check it against PennyLane's
`default.qubit`. The original PennyLane spike is kept in `spike/`. Device noise comes from Qiskit Aer with the offline calibration snapshots
of `ibm_kingston`, `ibm_fez`, and `ibm_marrakesh`, so no IBM Quantum account is required.
