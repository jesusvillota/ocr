from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

from paper_ocr.cli import extract_tables, find_marker, move_figures


def test_extract_tables() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        out_dir = Path(tmp) / "output"
        markdown, tables = extract_tables(
            "Before\n| Name | Value |\n| --- | --- |\n| A | 1 |\nAfter\n", out_dir
        )

        assert markdown == "Before\n\n[Table 1](tables/table_1.md)\n\nAfter\n"
        assert tables == [str(out_dir / "tables" / "table_1.md")]
        assert (out_dir / "tables" / "table_1.md").read_text() == (
            "| Name | Value |\n| --- | --- |\n| A | 1 |\n"
        )


def test_move_figures() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        source_dir = tmp_path / "source"
        source_dir.mkdir()
        (source_dir / "image.png").write_bytes(b"png")
        out_dir = tmp_path / "output"

        markdown, figures = move_figures("![Figure](image.png)", source_dir, out_dir)

        assert markdown == "![Figure](figures/figure_1.png)"
        assert figures == [str(out_dir / "figures" / "figure_1.png")]
        assert (out_dir / "figures" / "figure_1.png").read_bytes() == b"png"


def test_find_marker_uses_interpreter_bin() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        executable = Path(tmp) / "bin" / "python"
        marker = executable.parent / "marker_single"
        marker.parent.mkdir()
        executable.touch()
        marker.touch()

        with patch.object(sys, "executable", str(executable)):
            assert find_marker() == str(marker)


if __name__ == "__main__":
    test_extract_tables()
    test_move_figures()
    test_find_marker_uses_interpreter_bin()
