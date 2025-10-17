#!/usr/bin/env python3
"""
Build a single combined PDF from multiple Markdown files.
Usage:
  python scripts/build_prereg_pdf.py docs/prereg/H1*.md docs/prereg/H2*.md docs/prereg/H3*.md -o docs/prereg/Preregistration_Pack_v2.pdf
If pandoc is installed, it is used. Otherwise we fall back to a minimal renderer via FPDF.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
from datetime import date


def has_pandoc() -> bool:
    """Check if pandoc is available on the system."""
    return shutil.which("pandoc") is not None


def run_pandoc(md_files: list[str], out_pdf: str) -> None:
    """Run pandoc to convert markdown files to PDF."""
    cmd = ["pandoc", "-s", "-V", "geometry:margin=1in", "-o", out_pdf] + md_files
    subprocess.check_call(cmd)


def fallback_fpdf(md_files: list[str], out_pdf: str) -> None:
    """Fallback PDF generation using FPDF when pandoc is not available."""
    try:
        from fpdf import FPDF
    except ImportError:
        raise SystemExit("fpdf not installed. `pip install fpdf` or install pandoc.")

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)

    def add_h1(pdf, txt):
        """Add H1 heading to PDF."""
        pdf.set_font("Helvetica", "B", 18)
        pdf.ln(4)
        pdf.multi_cell(0, 10, txt.encode("latin-1", "replace").decode("latin-1"))
        pdf.ln(2)

    def add_h2(pdf, txt):
        """Add H2 heading to PDF."""
        pdf.set_font("Helvetica", "B", 14)
        pdf.ln(2)
        pdf.multi_cell(0, 8, txt.encode("latin-1", "replace").decode("latin-1"))

    def add_body(pdf, txt):
        """Add body text to PDF."""
        pdf.set_font("Helvetica", "", 11)
        pdf.multi_cell(0, 6, txt.encode("latin-1", "replace").decode("latin-1"))

    def add_code(pdf, code):
        """Add code block to PDF."""
        pdf.set_font("Courier", "", 9)
        pdf.set_fill_color(245, 245, 245)
        for line in code.splitlines() or [" "]:
            pdf.cell(
                0,
                5,
                txt=line[:120].encode("latin-1", "replace").decode("latin-1"),
                ln=1,
                fill=True,
            )
        pdf.ln(2)

    for path in md_files:
        pdf.add_page()
        with open(path, encoding="utf-8") as f:
            content = f.read()

        title = os.path.basename(path)
        add_h1(pdf, f"{title}  —  generated {date.today().isoformat()}")

        in_code = False
        code_buf = []
        for raw in content.splitlines():
            line = raw.rstrip("\n")
            if line.strip().startswith("```"):
                if in_code:
                    add_code(pdf, "\n".join(code_buf))
                    code_buf = []
                    in_code = False
                else:
                    in_code = True
                continue
            if in_code:
                code_buf.append(line)
                continue

            # very naive markdown parsing for # and ##
            if line.startswith("# "):
                add_h1(pdf, line[2:].strip())
            elif line.startswith("## "):
                add_h2(pdf, line[3:].strip())
            else:
                add_body(pdf, line)

        if in_code:
            add_code(pdf, "\n".join(code_buf))

    os.makedirs(os.path.dirname(out_pdf), exist_ok=True)
    pdf.output(out_pdf)


def main():
    """Main function to build PDF from markdown files."""
    p = argparse.ArgumentParser()
    p.add_argument("md", nargs="+", help="Markdown files, in order")
    p.add_argument("-o", "--out", required=True, help="Output PDF path")
    args = p.parse_args()

    # Ensure files exist
    for m in args.md:
        if not os.path.exists(m):
            raise FileNotFoundError(m)

    if has_pandoc():
        print("[INFO] Using pandoc for PDF generation")
        run_pandoc(args.md, args.out)
    else:
        print("[INFO] Using FPDF fallback for PDF generation")
        fallback_fpdf(args.md, args.out)

    print(f"[OK] Wrote {args.out}")


if __name__ == "__main__":
    main()
