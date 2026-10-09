# ROADMAP

## Experiment axes

Proposal commits to axis 1 (4 and 8 qubits) and shot noise; axes 2, device noise and 4 are
extensions, in that order.

1. **Circuit**: encoding reps x trainable layers, 4 then 8 qubits. Enough seeds and restarts to
   report a success rate with a range, not a single draw.
2. **Client**: batch size, unknown label (optimized jointly with the input), trained vs random θ.
3. **Device**: shot noise, then Aer noise models from `FakeKingston` / `FakeFez` /
   `FakeMarrakesh`. Does hardware noise act as a free defense?
4. **Attacker**: restarts budget, and Kumar et al.'s underparameterized attacker model.
5. **Report effective parameters** (nonzero-gradient count), not raw gate count, on every axis.
6. **Encoding family**: Kumar et al.'s "expressive" encoding is an exponential-frequency tower (m
   qubits per input), not linear re-uploading. Our re-uploading results test a different encoding,
   so the report must say so. A tower-encoding variant would test their regime directly.

`report/related-work-notes.md` holds the full-text extraction of the three core papers (claims,
threat models, qualifiers, three different definitions of "overparameterized").

## Deliverables

- Privacy map figure: recovery rate over (params, encoding reps), regimes marked.
- LaTeX report in IEEE QCE format (`report/`), written so it can go to QCE's QML track; in-class talk.

## Completed

- 2026-09-24: 4-qubit spike separates recovered / same-gradient-wrong-input / stuck; at 32 params
  the expressive encoding resists more often. First version's light-cone artifact found and fixed.
- 2026-10-08: JAX harness (#10); RQ1 (400 attacks, 4 and 8 qubits) and RQ2 (600 attacks, 1e2-1e6
  shots) run with 20 seeds. Encoding reps raise the stuck rate at fixed parameters; shot noise
  protects only 8-qubit circuits at ~100 shots. Report drafted from the results.
- 2026-10-09: one-page proposal sent to the instructor.
