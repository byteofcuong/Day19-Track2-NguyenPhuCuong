"""Package init — loads `.env` so the file the setup scripts write actually counts.

`setup-lite.sh` copies `.env.example` to `.env`, `setup-docker.sh` rewrites it to
switch on Qdrant server / Redis / bge-m3, and `.env.example` states that
`EMBEDDING_BACKEND` "IS read by app/embeddings.py". All true — except nothing
ever read the *file*. `app/embeddings.py`, `app/search.py` and the Feast config
call `os.getenv`, which sees the process environment only. Editing `.env` and
running `make api` therefore did nothing at all, silently.

Parsed here rather than with python-dotenv on purpose: dotenv arrives only as a
transitive dependency of `uvicorn[standard]` and is in no requirements file, so
depending on it would break the moment that extra changes. The format the lab
uses is `KEY=value` with `#` comments — twenty lines cover it.

Real environment variables always win: `EMBEDDING_BACKEND=bge-m3 make api` must
override the file, not the other way round.
"""
from __future__ import annotations

import os
from pathlib import Path

_ENV_FILE = Path(__file__).resolve().parent.parent / ".env"


def load_env(path: Path = _ENV_FILE, override: bool = False) -> dict[str, str]:
    """Load KEY=value pairs from `path` into os.environ. Returns what it set."""
    if not path.exists():
        return {}
    applied: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        # strip inline comments and surrounding quotes: `KEY=val   # note`
        value = value.split("#", 1)[0].strip().strip("'\"")
        if not key:
            continue
        if override or key not in os.environ:
            os.environ[key] = value
            applied[key] = value
    return applied


load_env()
