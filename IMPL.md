# IMPL

## Now

✅ Final paper (#29, PR #30): RQ1-RQ5 in `report/report.tex`, 7 pages, `/graphe:latex` CLEAN.

- Every number is a macro from `uv run python -m qflp.figures` (`report/tables/numbers.tex`);
  `qflp/macros.py` and `qflp/figures.py` raise `ClaimError` when an unnumbered sentence stops
  holding. `tests/test_report.py`: tables and macros regenerate byte-identical, every macro quoted,
  each guard fires on its own mutant (every guard switched off: 0 survivors).
- Citations: dokimasia lint 0 findings, verify 13 verified + jax2018 by hand, 14/14 rendered.
- Review: Copilot + agy lanes, five fresh Opus rounds, verify read clean.

Verify: `flock -o /tmp/fleet-heavy.lock uv run pytest -q`, then
`uv run --no-project python <graphe>/skills/latex/scripts/latex.py report/report.tex`.

## Next

📋 The "Live lab" talk deck (fleet queue 2b(ii)): 10-12 slides + script in the talks repo, numbers
from `report/tables/numbers.tex` only; the in-browser toy demonstrates the mechanism and says so.
Talks from Tue 12-01, 13-15 min.

📋 Later, if time: Aer fake-backend noise; tower-encoding variant (Kumar et al.'s regime). Open:
per-input error inside an ambiguous batch; the 2^B label-enumeration attacker; more training
examples per seed (would test the curves D = 2 lead, and separate fit from frequency); 50-restart
runs on the trained 8-qubit 4×1 circuit.
