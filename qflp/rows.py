"""Read and repair sweep JSONL files. Imports no JAX, so the sweep's parent process stays light."""

import json
import os
from pathlib import Path

SETTING = ("n_qubits", "reps", "layers", "shots", "batch", "restarts")
KEY = (*SETTING, "seed")  # one row per run


def read_rows(path: Path, repair: bool = False) -> list[dict]:
    """Every row of a sweep file, one JSON object per line.

    A killed sweep can leave its last line unterminated. If that line is a complete row it is kept;
    if it is a cut-off row (starts with "{" but does not parse) it is skipped. With `repair`, the
    file is fixed in place so the next append starts on a fresh line: a newline is added after a
    complete row, or the cut-off row is truncated away. Truncating in place keeps the same file, so
    a symlink target, its permissions and any other open appender are unaffected. Every other
    line that is not a JSON object raises, so a file that is not a sweep file is never changed.
    """
    data = path.read_bytes()
    lines = data.decode().splitlines(keepends=True)
    rows = []
    for i, line in enumerate(lines):
        last_unterminated = i == len(lines) - 1 and not line.endswith(("\n", "\r"))
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            row = None
        if isinstance(row, dict):
            rows.append(row)
            if last_unterminated and repair:
                with path.open("ab") as fh:
                    fh.write(b"\n")
            continue
        if last_unterminated and line.lstrip().startswith("{"):
            if repair:
                os.truncate(path, data.rfind(b"\n") + 1)
            break
        hint = " (blank line)" if not line.strip() else ""
        raise ValueError(f"{path}:{i + 1} is not a sweep row{hint}")
    return rows
