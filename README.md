# ocr

Convert a PDF paper into three parts:

- `paper.md` — the full text with equations as LaTeX (`$$` blocks and inline `$…$`).
- `figures/` — every figure extracted as a PNG.
- `tables/` — every table extracted as its own `table_N.md`, referenced inline from `paper.md`.

Powered by [marker](https://github.com/datalab-to/marker) (state-of-the-art PDF → markdown). Runs locally on Apple Silicon (MPS) or CUDA; no GPU required.

## Setup

Requires [uv](https://docs.astral.sh/uv/) and ~2 GB of disk for the model weights (downloaded automatically on first run).

```bash
uv venv --python 3.12
uv pip install marker-pdf
```

## Usage

```bash
.venv/bin/python ocr.py path/to/paper.pdf
```

Output lands in `papers/<paper-stem>/`:

```
papers/paper/
├── paper.md
├── figures/
│   ├── figure_1.png
│   └── ...
└── tables/
    ├── table_1.md
    └── ...
```

### Options

```bash
python ocr.py <input.pdf> [--out papers] [--use-llm] [--force-ocr] [--pages 0,5-10] [--name custom]
```

- `--out` — output root (default `papers`).
- `--use-llm` — boost accuracy on messy tables and inline math via an LLM. Off by default (free/local). Set `GEMINI_API_KEY` or `OPENAI_API_KEY` in your env; Gemini is the default backend.
- `--force-ocr` — re-OCR the whole PDF. Off by default (born-digital PDFs use fast, accurate text extraction). Turn it on for scanned/image PDFs, or when you want inline math converted to LaTeX throughout.
- `--pages` — comma-separated page range, e.g. `0,5-10,20`. Defaults to the whole document.
- `--name` — override the output folder name (defaults to the PDF filename stem).

Block equations are converted to LaTeX (`$$…$$`) regardless of this flag; `--force-ocr` additionally converts inline math to `$…$`.

## Example

See `examples/` for a converted paper (output only, no source PDF). Run the same on your own PDF with the one-liner above.

## License

GPL-3.0. Inherited from `marker-pdf`, which is GPL-3.0 (model weights use a modified OpenRAIL-M license — free for research, personal use, and startups under $2M funding/revenue). See `LICENSE`.
