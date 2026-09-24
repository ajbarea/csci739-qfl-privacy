# IMPL

## Now

Spike done (`spike/gradient_inversion.py`, results `spike/results-2026-09-24.txt`). 4 qubits, one
sample, known label, untrained θ, 5 seeds x 5 L-BFGS-B restarts:

- 8 params: gradient matched to ~1e-17 but input not recovered. Privacy here is
  non-identifiability, not hardness.
- 32 params: 9/10 exact, 1 at recon error 0.017.
- 128 params: 4/5 exact; one run stuck at match loss 1e+01, the regime Kumar et al. predict.

`tests/test_spike.py` pins the 8-param and 32-param (reps=4) cells at seed 0.

## Next

📋 Submit `proposal.md` on myCourses before 2026-10-09 23:59.
