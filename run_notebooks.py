"""Execute every lab notebook headless — the Windows-friendly `make notebooks`.

The Makefile hard-codes `.venv/bin/...`, which only exists on macOS/Linux; on
Windows the console scripts live in `.venv/Scripts/`. This script resolves the
venv the same way on both, and — importantly — puts that directory on PATH
before running nbconvert. NB3 shells out to `uvicorn` and NB4 to the `feast`
CLI; without PATH they die with FileNotFoundError inside the kernel.

    python run_notebooks.py            # all notebooks
    python run_notebooks.py 03 07      # just those, by number prefix

Exits non-zero if any notebook fails, so it can gate a commit.
"""
from __future__ import annotations

import glob
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BIN = ROOT / ".venv" / ("Scripts" if os.name == "nt" else "bin")
EXE = ".exe" if os.name == "nt" else ""
TIMEOUT_S = 1800  # NB1 embeds 1000 docs on a cold model cache


def tool(name: str) -> str:
    path = BIN / f"{name}{EXE}"
    if not path.exists():
        sys.exit(f"{path} not found — run `bash setup-lite.sh` first.")
    return str(path)


def main(argv: list[str]) -> int:
    env = dict(os.environ)
    env["PATH"] = f"{BIN}{os.pathsep}{env.get('PATH', '')}"
    env.setdefault("PYTHONIOENCODING", "utf-8")

    # Numbered notebooks only. `_setup.py` is an import helper, not a notebook:
    # converting it produces a `_setup.ipynb` that fails on execute, which is
    # exactly why setup-lite.sh and the Makefile both glob `[0-9]*.py` too.
    print("Syncing .py -> .ipynb (jupytext)...")
    subprocess.run([tool("jupytext"), "--to", "notebook", "--update",
                    *sorted(glob.glob("notebooks/[0-9]*.py"))],
                   cwd=ROOT, env=env, check=False)

    wanted = tuple(argv) if argv else None
    notebooks = [n for n in sorted(glob.glob("notebooks/[0-9]*.ipynb"))
                 if wanted is None or Path(n).name.startswith(wanted)]
    if not notebooks:
        sys.exit(f"No notebooks matched {argv}")

    failed: list[str] = []
    for nb in notebooks:
        print(f"\nExecuting {nb} ...", flush=True)
        res = subprocess.run(
            [tool("jupyter"), "nbconvert", "--to", "notebook", "--execute",
             "--inplace", nb, f"--ExecutePreprocessor.timeout={TIMEOUT_S}"],
            cwd=ROOT, env=env,
        )
        status = "PASS" if res.returncode == 0 else "FAIL"
        print(f"{status}: {nb}")
        if res.returncode != 0:
            failed.append(nb)

    print("\n" + "-" * 60)
    print(f"{len(notebooks) - len(failed)}/{len(notebooks)} notebooks passed")
    if failed:
        print("failed: " + ", ".join(failed))
        return 1
    print("Next: python scripts/make_screenshots.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
