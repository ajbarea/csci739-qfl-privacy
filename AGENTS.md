# csci739-qfl-privacy

AJ's CSCI-739 (Quantum Machine Learning, Dr. Ryan Vogt, RIT Fall 2026) final project: **when do
variational quantum circuits keep a federated client's training data private?** Course notes and
homework live in the sibling `csci739-quantum-machine-learning` repo; read its `AGENTS.md` for
course facts.

## Deliverables

| | |
|---|---|
| Proposal | myCourses assignment, due 2026-10-09 23:59 (week 7). Text in `proposal.md` |
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

- **PennyLane `default.qubit`, `diff_method="backprop"`** for the attack. It needs the gradient of a
  gradient-matching loss (second-order autodiff); Qiskit's parameter-shift path made the first spike
  impractically slow.
- **Qiskit Aer + `qiskit_ibm_runtime.fake_provider`** (`FakeKingston`, `FakeFez`, `FakeMarrakesh`)
  for calibrated Heron r2 noise, offline, no IBM account needed.
- `uv run python spike/gradient_inversion.py` reproduces the spike.

## Spike (2026-09-24)

`spike/results-2026-09-24.txt`. 4 qubits, one sample, known label, untrained θ, 5 seeds:

- 8 params: gradient matched to ~1e-17 yet input **not** recovered. Many inputs give the same
  gradient, so privacy here comes from non-identifiability, not hardness.
- 32 params (either more reps or more layers): input recovered exactly in 10/10.
- 128 params: 4/5 recovered, one stuck in a local minimum (match loss 1e+01) with 5 restarts.

Not yet tested: batches, unknown labels, trained θ, 8+ qubits, shot noise, device noise.
