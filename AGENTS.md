# csci739-qfl-privacy

AJ's CSCI-739 (Quantum Machine Learning, Dr. Ryan Vogt, RIT Fall 2026) final project: **when do
variational quantum circuits keep a federated client's training data private?** Public sister repo.
Course notes and homework live in the private `~/ajsoftworks/classes/csci739-quantum-machine-learning`;
read its `AGENTS.md` for course facts. **Never copy lecture transcripts or classmates' names here**:
this repo is public, the class repo is not.

## Deliverables

| | |
|---|---|
| Proposal | myCourses assignment, due 2026-10-09 23:59 (week 7). `proposal/proposal.tex`, built with `/techne:latex` |
| Report | LaTeX, methodology + findings (syllabus requirement) |
| Talk | In class, weeks 12-14 |

Grade weight: 25% of the course.

## The question

Kumar et al. 2023 (arXiv 2309.13002) argue expressive encodings make gradient inversion on VQCs
intractable (spurious local minima in high-degree Chebyshev polynomial systems). Papadopoulos et al.
2025 (arXiv 2504.12806), same JPMorgan group, reconstruct inputs once the model is overparameterized.
The project maps where each holds across encoding reps, trainable layers, batch size, and hardware
noise.

## Toolchain

- uv, ruff (format + check), ty, pytest. CI runs exactly those on 3.12-3.14.
- **PennyLane `default.qubit`, `diff_method="backprop"`** for the attack. It needs the gradient of a
  gradient-matching loss (second-order autodiff); Qiskit's parameter-shift path made the first spike
  impractically slow.
- **Qiskit Aer + `qiskit_ibm_runtime.fake_provider`** (`FakeKingston`, `FakeFez`, `FakeMarrakesh`)
  for calibrated Heron r2 noise, offline, no IBM account needed.
- `pennylane.numpy` is generated dynamically, so `ty` cannot see its functions: prefer array methods
  (`(d**2).sum()`) over `pnp.sum`.
- **Check the light cone before reading a result.** Every input must reach the gradient
  (`test_every_input_reaches_the_gradient`); a single-qubit observable silently dropped three of four
  inputs in the first spike. Compare inputs mod 2π, never through cos, which also folds x onto -x.
- Simulations are CPU-only. AJ's GPU may be busy with other experiments; large sweeps can go
  overnight or to the RIT cluster.

State and next steps: `IMPL.md`. Long view: `ROADMAP.md`.
