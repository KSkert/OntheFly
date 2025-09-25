from __future__ import annotations
import re

def _parse_step(path: str) -> int:
    m = re.search(r"step(\d+)\.pt$", path)
    if m:
        try:
            return int(m.group(1))
        except ValueError:
            return 0
    return 0
