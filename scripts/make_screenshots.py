"""Render executed notebook outputs into submission/screenshots/*.png.

Why a script and not a real screen capture: the deliverable evidence is text
(counts, latency tables, precision tables), and a script is reproducible --
re-run it after re-executing the notebooks and the images stay in sync with
what the notebooks actually printed. Run it AFTER `make notebooks`.

    python scripts/make_screenshots.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
NB_DIR = ROOT / "notebooks"
OUT_DIR = ROOT / "submission" / "screenshots"

# A terminal-ish dark theme; readable at 100% zoom in a browser.
BG = (24, 26, 33)
FG = (222, 226, 235)
DIM = (128, 136, 155)
ACCENT = (126, 200, 227)
TITLEBAR = (38, 41, 51)
FONT_SIZE = 16
LINE_H = 22
PAD = 24
MAX_LINES = 60          # split into ..._2.png, ..._3.png beyond this
WRAP = 104              # characters per line before hard-wrapping

# Windows/macOS/Linux monospace fonts that cover Vietnamese diacritics.
FONT_CANDIDATES = [
    "C:/Windows/Fonts/consola.ttf",
    "C:/Windows/Fonts/CascadiaMono.ttf",
    "/System/Library/Fonts/Menlo.ttc",
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
]

ANSI = re.compile(r"\x1b\[[0-9;]*[a-zA-Z]")
# tqdm/ipywidgets noise that adds nothing to the evidence
NOISE = ("TqdmWarning", "from .autonotebook import", "IProgress not found")


def load_font(size: int) -> ImageFont.FreeTypeFont:
    for path in FONT_CANDIDATES:
        if Path(path).exists():
            try:
                return ImageFont.truetype(path, size)
            except OSError:
                continue
    return ImageFont.load_default()


def notebook_text(nb_path: Path) -> list[str]:
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    lines: list[str] = []
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        chunk: list[str] = []
        for out in cell.get("outputs", []):
            kind = out.get("output_type")
            if kind == "stream":
                chunk += "".join(out.get("text", "")).splitlines()
            elif kind in ("execute_result", "display_data"):
                chunk += "".join(out.get("data", {}).get("text/plain", "")).splitlines()
            elif kind == "error":
                chunk += [f"!! {out.get('ename')}: {out.get('evalue')}"]
        chunk = [ANSI.sub("", l).rstrip() for l in chunk]
        chunk = [l for l in chunk if not any(n in l for n in NOISE)]
        while chunk and not chunk[0].strip():
            chunk.pop(0)
        while chunk and not chunk[-1].strip():
            chunk.pop()
        if chunk:
            if lines:
                lines.append("")
            lines += chunk
    return lines


def wrap(lines: list[str], width: int) -> list[str]:
    out: list[str] = []
    for line in lines:
        if len(line) <= width:
            out.append(line)
            continue
        while line:
            out.append(line[:width])
            line = line[width:]
    return out


def render(title: str, lines: list[str], dest: Path, font, title_font) -> None:
    w = PAD * 2 + max(int(font.getlength("M")) * WRAP, int(title_font.getlength(title)))
    h = PAD * 2 + 40 + LINE_H * max(len(lines), 1)
    img = Image.new("RGB", (w, h), BG)
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, w, 40], fill=TITLEBAR)
    for i, c in enumerate(((255, 95, 86), (255, 189, 46), (39, 201, 63))):
        d.ellipse([16 + i * 20, 15, 26 + i * 20, 25], fill=c)
    d.text((86, 12), title, font=title_font, fill=ACCENT)
    y = 40 + PAD
    for line in lines:
        color = DIM if line.startswith(("  ", "\t")) and not line.strip() else FG
        d.text((PAD, y), line, font=font, fill=color)
        y += LINE_H
    dest.parent.mkdir(parents=True, exist_ok=True)
    img.save(dest)


def main() -> int:
    font, title_font = load_font(FONT_SIZE), load_font(FONT_SIZE - 1)
    notebooks = sorted(NB_DIR.glob("[0-9]*.ipynb"))
    if not notebooks:
        print("No executed notebooks found — run `make notebooks` first.")
        return 1
    made, skipped = 0, []
    for nb in notebooks:
        lines = wrap(notebook_text(nb), WRAP)
        if not lines:
            skipped.append(nb.name)
            continue
        pages = [lines[i:i + MAX_LINES] for i in range(0, len(lines), MAX_LINES)]
        for n, page in enumerate(pages, start=1):
            suffix = "" if len(pages) == 1 else f"_{n}"
            title = f"{nb.stem}.ipynb — output{'' if len(pages) == 1 else f' ({n}/{len(pages)})'}"
            dest = OUT_DIR / f"{nb.stem}{suffix}.png"
            render(title, page, dest, font, title_font)
            print(f"  wrote {dest.relative_to(ROOT)}  ({len(page)} lines)")
            made += 1
    if skipped:
        print(f"\nNo output cells (not executed?): {', '.join(skipped)}")
    print(f"\n{made} screenshot(s) in {OUT_DIR.relative_to(ROOT)}")
    return 1 if skipped else 0


if __name__ == "__main__":
    sys.exit(main())
