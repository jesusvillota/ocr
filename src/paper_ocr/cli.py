#!/usr/bin/env python3
"""OCR a PDF paper to markdown + figures/ + tables/.

Wrapper around `marker_single` (marker-pdf) that reorganizes its output into
papers/<name>/{paper.md, figures/, tables/}.

- figures/ : every image marker extracted, renamed figure_N.<ext>, links rewritten.
- tables/  : every markdown table block, extracted to table_N.md and replaced
             inline with [Table N](tables/table_N.md).
- paper.md : the rest, with LaTeX math ($$, $...$) preserved as marker emits it.

`--force_ocr` is always passed (clean inline math, handles scanned PDFs).
"""
from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path

IMG_RE = re.compile(r"!\[([^\]]*)\]\(([^)]+)\)")
# A markdown table separator row: |---|:--:|---| etc., pipes between cells, edge pipes optional.
# ponytail: regex parser, not a full MD AST — fine because the separator row is the unambiguous
# table signal; a lone "a | b" in prose is never followed by a separator, so it won't false-fire.
SEP_RE = re.compile(r"^\s*\|?\s*:?-+:?\s*(\|\s*:?-+:?\s*)*\|?\s*$")
ROW_RE = re.compile(r"^\s*\|?.*\|.*\|?\s*$")  # a line with at least 2 pipes-ish


def find_marker() -> str:
    candidate = Path(sys.executable).parent / "marker_single"
    if candidate.exists():
        return str(candidate)
    found = shutil.which("marker_single")
    if found:
        return found
    sys.exit("error: marker_single not found. Install marker-pdf in this environment.")


def run_marker(pdf: Path, staging: Path, force_ocr: bool, use_llm: bool, pages: str | None, env: dict[str, str]) -> None:
    cmd = [
        find_marker(), str(pdf),
        "--output_dir", str(staging),
    ]
    if force_ocr:
        cmd.append("--force_ocr")  # re-OCR everything (scanned/image PDFs, inline math as LaTeX)
    if pages:
        cmd += ["--page_range", pages]
    if use_llm:
        cmd += ["--use_llm"]
    print(f"running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True, env=env)


def find_md(staging: Path) -> Path:
    mds = list(staging.rglob("*.md"))
    if not mds:
        sys.exit(f"error: no markdown produced in {staging}")
    # marker writes one <stem>.md; pick the largest to ignore stray files
    return max(mds, key=lambda p: p.stat().st_size)


def move_figures(md_text: str, md_dir: Path, out_dir: Path) -> tuple[str, list[str]]:
    """Move every referenced image into figures/, rename, rewrite links.
    Returns (new_md_text, list_of_figure_paths)."""
    figs_dir = out_dir / "figures"
    figs_dir.mkdir(parents=True, exist_ok=True)

    refs: list[tuple[str, str]] = []  # (alt, raw_path)
    seen_paths: dict[str, str] = {}   # raw_path -> figure_N.<ext>

    def repl(m: re.Match) -> str:
        alt, raw = m.group(1), m.group(2)
        # strip any query/hash, keep relative as-is
        src = raw.split("?")[0].split("#")[0]
        src_path = (md_dir / src).resolve() if not Path(src).is_absolute() else Path(src)
        if src not in seen_paths:
            if not src_path.exists():
                # leave link untouched if file missing (marker sometimes embeds base64 or remote)
                return m.group(0)
            n = len(seen_paths) + 1
            ext = src_path.suffix or ".png"
            new_name = f"figure_{n}{ext}"
            seen_paths[src] = new_name
            shutil.copy2(src_path, figs_dir / new_name)
        return f"![{alt}](figures/{seen_paths[src]})"

    new_md = IMG_RE.sub(repl, md_text)

    # also sweep any leftover image files marker dropped in staging that weren't linked
    for img in md_dir.glob("*"):
        if img.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"} and img.name not in {
            Path(p).name for p in seen_paths
        }:
            n = len(seen_paths) + 1
            new_name = f"figure_{n}{img.suffix}"
            shutil.copy2(img, figs_dir / new_name)
            seen_paths[str(img)] = new_name

    fig_paths = sorted(str(figs_dir / v) for v in seen_paths.values())
    return new_md, fig_paths


def extract_tables(md_text: str, out_dir: Path) -> tuple[str, list[str]]:
    """Pull every markdown table block into tables/table_N.md; replace inline
    with [Table N](tables/table_N.md). Returns (new_md_text, table_paths)."""
    tables_dir = out_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    lines = md_text.split("\n")
    out: list[str] = []
    table_paths: list[str] = []
    i = 0
    n = 0
    while i < len(lines):
        # a table needs: header row, separator row, >=1 body row
        if i + 2 <= len(lines) and SEP_RE.match(lines[i + 1]) and _is_row(lines[i]):
            block_start = i
            i += 2  # skip header + sep
            while i < len(lines) and _is_row(lines[i]) and lines[i].strip() != "":
                i += 1
            block = lines[block_start:i]
            n += 1
            tname = f"table_{n}.md"
            (tables_dir / tname).write_text("\n".join(block) + "\n", encoding="utf-8")
            table_paths.append(str(tables_dir / tname))
            out.append(f"\n[Table {n}](tables/{tname})\n")
        else:
            out.append(lines[i])
            i += 1
    return "\n".join(out), table_paths


def _is_row(line: str) -> bool:
    s = line.strip()
    if not s:
        return False
    # a table row has at least one pipe (edge-piped or inline-piped)
    return "|" in s and not SEP_RE.match(line)


def reorganize(staging: Path, out_dir: Path) -> dict[str, object]:
    md_path = find_md(staging)
    md_text = md_path.read_text(encoding="utf-8")
    md_dir = md_path.parent

    out_dir.mkdir(parents=True, exist_ok=True)
    md_text, figs = move_figures(md_text, md_dir, out_dir)
    md_text, tables = extract_tables(md_text, out_dir)
    (out_dir / "paper.md").write_text(md_text, encoding="utf-8")

    eqs = md_text.count("$$")
    return {"figures": len(figs), "tables": len(tables), "equation_blocks": eqs // 2}


def main() -> int:
    p = argparse.ArgumentParser(description="OCR a PDF to markdown + figures + tables.")
    p.add_argument("pdf", type=Path, help="input PDF")
    p.add_argument("--out", type=Path, default=Path("papers"), help="output root (default: papers)")
    p.add_argument("--use-llm", action="store_true", help="use an LLM to boost accuracy (needs API key)")
    p.add_argument("--force-ocr", action="store_true", help="re-OCR the whole PDF (scanned/image PDFs; inline math as LaTeX)")
    p.add_argument("--pages", default=None, help='page range, e.g. "0,5-10,20"')
    p.add_argument("--name", default=None, help="output folder name (default: PDF stem)")
    p.add_argument("--keep-staging", action="store_true", help="keep raw marker output next to results")
    args = p.parse_args()

    if not args.pdf.exists():
        sys.exit(f"error: {args.pdf} not found")

    import os
    env = dict(os.environ)
    if args.use_llm:
        if not (env.get("GEMINI_API_KEY") or env.get("OPENAI_API_KEY")):
            sys.exit("error: --use-llm requires GEMINI_API_KEY or OPENAI_API_KEY in env")

    name = args.name or args.pdf.stem
    out_dir = (args.out / name).resolve()
    staging = out_dir.parent / f".{name}_staging"

    if staging.exists():
        shutil.rmtree(staging)
    run_marker(args.pdf, staging, args.force_ocr, args.use_llm, args.pages, env)

    if out_dir.exists():
        shutil.rmtree(out_dir)
    stats = reorganize(staging, out_dir)

    if args.keep_staging:
        staging.rename(out_dir.parent / f".{name}_staging_kept")
    else:
        shutil.rmtree(staging)

    print(f"\ndone → {out_dir}")
    print(f"  {stats['figures']} figures, {stats['tables']} tables, {stats['equation_blocks']} equation blocks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
