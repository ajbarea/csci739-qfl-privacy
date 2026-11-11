# IMPL

## Now

Final paper (#29, PR #30, branch `paper/final`).

- ✅ RQ5 trained circuits merged earlier (#16).
- ✅ Every number in `report/report.tex` is a macro from `uv run python -m qflp.figures`
  (`report/tables/numbers.tex`). `qflp/macros.py` holds the method, RQ1, RQ2 and budget numbers and
  raises `ClaimError` when an unnumbered sentence stops holding. Typed numbers are setting labels,
  cited papers' figures and math only.
- ✅ `tests/test_report.py`: tables and macros regenerate byte-identical; every macro quoted, every
  quoted macro generated; each claim guard fires on a mutant.
- ✅ Build: `/graphe:latex` CLEAN, 7 of 12 pages (budget declared in the source); abstract ~249 words.
- ✅ Citations: `dokimasia verify` 13 verified, jax2018 checked by hand against the JAX README.
- 📋 Review: Copilot + agy lanes on the PDF text vs results, then fresh Opus readers to the first
  clean round (ceiling 5, one verify read after the last fixes).
- 📋 Merge #30 when clean and CI green; PDF to `Desktop\To review\CSCI-739 QML final paper.pdf`.

Verify: `uv run pytest -q` (wrap in the fleet heavy lock), then
`uv run --no-project python <graphe>/skills/latex/scripts/latex.py report/report.tex`.

## Next

📋 The "Live lab" talk deck (fleet queue 2b(ii)): 10-12 slides + script in the talks repo, numbers
from `report/tables/numbers.tex` only; the in-browser toy demonstrates the mechanism and says so.

📋 Later, if time: Aer fake-backend noise; tower-encoding variant (Kumar et al.'s regime). Open:
per-input error inside an ambiguous batch; the 2^B label-enumeration attacker; more training
examples per seed (would test the curves D = 2 lead).
