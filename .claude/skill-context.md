# skill-context: csci739-qfl-privacy

Repo-specific facts the techne skills read. Logic lives in the skills; only facts belong here.

## repo

- name: csci739-qfl-privacy
- kind: research code for a course final project (CSCI-739, RIT Fall 2026). Public.
- default_branch: main
- description: gradient-inversion attacks on variational quantum circuits in federated learning,
  mapping when the circuit keeps client data private
- language: Python (>=3.12,<3.15); CI matrix covers 3.12, 3.13, 3.14
- toolchain: uv (canonical); ruff (format + lint), ty (types), pytest
- package_root: `spike/` (flat, no src-layout yet)
- runner: none. No Makefile and no `logs/dev-<ts>-*.log` archive convention, so the audit's
  log-reconciliation phase is N/A.
- has: PennyLane (attack), Qiskit Aer + qiskit-ibm-runtime fake backends (device noise). No
  Docker, no docs site.

## audit

### Phase 1 — Setup

- `uv sync --all-groups`

### Phase 2 — Fix (one-way door)

- `uv run ruff format .`

### Phase 3 — Lint

- `uv run ruff format --check .`
- `uv run ruff check .`
- `uv run ty check`

### Phase 4 — Test

- `uv run pytest -q` (about 30 s; runs two attack cells)

### do_not_run (long-running)

- `uv run python spike/gradient_inversion.py` (full spike, several minutes of CPU)
