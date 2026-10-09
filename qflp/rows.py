"""Read and repair sweep JSONL files. Imports no JAX, so the sweep's parent process stays light."""

import json
import os
from pathlib import Path

SETTING = ("n_qubits", "reps", "layers", "shots", "batch", "restarts")
KEY = (*SETTING, "seed")  # one row per run
# Every row starts with the job's first field, so a cut-off row is a prefix of this or extends it.
ROW_START = '{"n_qubits": '


def read_rows(path: Path, repair: bool = False) -> list[dict]:
    """Every row of a sweep file, one JSON object per line.

    A killed sweep can leave its last line unterminated. If that line is a complete row it is kept;
    if it is a cut-off row (begins like ROW_START but does not parse) it is skipped. With `repair`,
    the file is fixed in place so the next append starts on a fresh line: a newline is added after
    a complete row, or exactly the cut-off row's bytes are truncated away. Truncating in place keeps
    the same file, so a symlink and its target's permissions survive. Repair assumes no one else is
    appending; `qflp.sweep` holds a lock on the file for that. Any other line that is not a row (a
    JSON object with every KEY field) raises, so a file that is not a sweep file is never changed.
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
        if isinstance(row, dict) and all(k in row for k in KEY):
            rows.append(row)
            if last_unterminated and repair:
                with path.open("ab") as fh:
                    fh.write(b"\n")
            continue
        head = line.lstrip()
        cut_off = head and (head.startswith(ROW_START) or ROW_START.startswith(head))
        if last_unterminated and cut_off and row is None:
            if repair:
                os.truncate(path, len(data) - len(line.encode()))
            break
        hint = " (blank line)" if not line.strip() else ""
        raise ValueError(f"{path}:{i + 1} is not a sweep row{hint}")
    return rows
