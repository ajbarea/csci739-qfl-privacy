# IMPL

## Now

Spike done (`spike/gradient_inversion.py`, results `spike/results-2026-09-24.txt`). 4 qubits,
observable = mean Z over all qubits, one sample, known label, untrained θ, 5 seeds x 5 L-BFGS-B
restarts, error = circular distance mod 2π:

- 1x1 (8 params): 1 exact, 1 within 0.07, 2 found a different input with the identical gradient
  (match ~1e-13: the gradient system has several exact preimages), 1 stuck.
- 1x4 (32): 5/5 recovered.
- 4x1 (32): 3/5 recovered, 2 stuck. Same parameter count, more expressive encoding, harder attack.
- 4x4 (128): 4/5 recovered, 1 stuck.

The first version measured Z on qubit 0 only. Its CNOT chain points away from qubit 0, so at 1x1
three of four inputs never reached the gradient and the "ambiguous" regime was an artifact.
`test_every_input_reaches_the_gradient` guards against that class of bug.

Parameter counts include the final block's RZ gates, which sit before a Z-diagonal measurement and
always have zero gradient.

## Next

📋 Upload `proposal/proposal.pdf` to the myCourses final-project-proposal assignment (Vogt asked for it 2026-09-24; due 2026-10-09 23:59).
