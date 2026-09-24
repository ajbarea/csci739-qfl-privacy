# ROADMAP

## Experiment axes

1. **Circuit**: encoding reps x trainable layers, 4 then 8 qubits. Enough seeds and restarts to
   report a success rate with a range, not a single draw.
2. **Client**: batch size, unknown label (optimized jointly with the input), trained vs random θ.
3. **Device**: shot noise, then Aer noise models from `FakeKingston` / `FakeFez` /
   `FakeMarrakesh`. Does hardware noise act as a free defense?
4. **Attacker**: restarts budget, and Kumar et al.'s underparameterized attacker model.
5. **Report effective parameters** (nonzero-gradient count), not raw gate count, on every axis.

## Deliverables

- Privacy map figure: recovery rate over (params, encoding reps), regimes marked.
- LaTeX report, in-class talk.

## Completed

- 2026-09-24: 4-qubit spike separates recovered / same-gradient-wrong-input / stuck; at 32 params
  the expressive encoding resists more often. First version's light-cone artifact found and fixed.
