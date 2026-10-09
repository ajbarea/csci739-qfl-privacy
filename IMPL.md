# IMPL

## Now

Experiment harness (#10) on `feat/experiment-harness`:

- `qflp/circuit.py`: JAX statevector simulator of the spike's circuit, `lax.scan` over blocks, so it
  compiles once per shape (<10 s at 8 qubits, `qflp.compile_time`). Pinned to PennyLane `default.qubit` at 1e-12.
- `qflp/attack.py`: L-BFGS-B gradient matching, 10 restarts; outcomes recovered (≤0.05 rad, mod
  2π) / close (≤0.2 rad) / ambiguous (further away, fits as well as the true input) / stuck. The attacker keeps the lowest-loss
  restart.
- `qflp/noise.py`: finite-shot parameter-shift client gradient.
- `qflp/sweep.py` (resumable JSONL), `qflp/summarize.py`, `qflp/figures.py` (all report figures
  and generated table rows).

Results (`results/`, 20 seeds per setting):

- `rq1.jsonl`, 400 attacks: every 1×ℓ circuit with ℓ ≥ 2 recovered 20/20 at 4 and 8 qubits.
  At fixed parameters, more encoding reps → more stuck; 8 qubits, 120 effective params: 20, 18,
  8, 2 recovered for r = 1, 2, 4, 8. r = ℓ = 1 gives a different exact preimage (16/20, 20/20).
- `rq2.jsonl`, 600 attacks, independent shot-noise stream per run: at 4 qubits, 100 shots still
  leaves 10-17/20 within 0.2 rad; error falls ~S^-1/2. At 8 qubits, 100 shots protects (median
  1.0-2.4 rad, ≤1/20 within 0.2 rad); by 1e4 shots 1×4 is back to 20/20 within 0.2 rad. 4-qubit
  4×1 is stuck on the same 4 seeds at every shot count; 8-qubit 4×1 stays stuck on 11-12.
- `budget.jsonl`, 80 attacks with 50 restarts on 8-qubit 2×1, 4×1, 4×2, 8×1: recovered seeds rise
  7→18, 8→17, 8→18, 2→5; per-restart success 7.9, 4.8, 6.2, 1.1%. The first 10 restarts repeat
  RQ1 exactly.

`report/report.tex` is drafted from these, every section filled; tables are generated rows.

## Next

📋 Extensions in ROADMAP order: batch size and unknown label, trained θ, Aer fake-backend noise,
tower-encoding variant (Kumar et al.'s regime).
