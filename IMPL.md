# IMPL

## Now

Client axis (#13) on `feat/client-axis`:

- `qflp/sweep.py --label fixed|known|unknown`: `fixed` is y = 1 (all earlier sweeps; old rows load
  as `fixed` via `rows.DEFAULTS`); `known`/`unknown` draw y ∈ {-1, +1} from a child seed stream, so
  both modes attack the same client from the same starts.
- `qflp/attack.py`: label-free loss = distance from g to span{∇f(x̂_i)} (B×B ridge solve,
  `RIDGE` = 1e-12 relative); the attacker gets placeholder labels.
- `qflp/figures.py`: `batch_map` (recovered vs N·B / P_eff), `batch_table`, `label_table` with an
  exact McNemar p on the paired seeds.

Results (`results/`, 20 seeds, exact gradients, 10 restarts):

- `rq3.jsonl`, 240 attacks, B ∈ {2, 4, 8} on 4/8-qubit 1×2 and 1×4 (B = 1 from RQ1): ≥12/20
  recovered whenever N·B < P_eff, 0/20 and all ambiguous whenever N·B > P_eff (worst input a median
  1.1-2.5 rad off). Below the line, 8-qubit misses are stuck (6 at 1×2 B=2, 8 at 1×4 B=4).
- `rq4.jsonl`, 320 attacks, B ∈ {1, 2}, labels known vs unknown on the same clients: B = 1 within
  one seed; largest gap 8-qubit 1×2 B=2, 13 vs 6 recovered, McNemar p = 0.065.

`report/report.tex` has RQ3 and RQ4 sections; tables and the batch figure are generated.

## Next

📋 Extensions in ROADMAP order: trained θ, Aer fake-backend noise, tower-encoding variant (Kumar
et al.'s regime). Open from RQ3/RQ4: per-input error inside an ambiguous batch; the 2^B
label-enumeration attacker.
