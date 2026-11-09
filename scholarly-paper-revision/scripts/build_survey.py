#!/usr/bin/env python3
"""Build a LaTeX root and check resolved references and optional PDF page bounds.

Visual inspection and scholarly review are separate from this build check.
Uses only the Python standard library and installed TeX/Poppler command-line tools.
"""

import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path


def page_count(pdf_path):
    result = subprocess.run(
        ["pdfinfo", str(pdf_path)], capture_output=True, text=True, check=True
    )
    match = re.search(r"^Pages:\s+(\d+)\s*$", result.stdout, re.MULTILINE)
    if not match:
        raise ValueError("Could not read the PDF page count from pdfinfo.")
    return int(match.group(1))


def unresolved_messages(log_text):
    patterns = (
        r"Undefined control sequence",
        r"(?:Citation|Reference|Hyper reference)\s+[^\n]*undefined",
        r"There were undefined (?:references|citations)",
        r"Please\s+\(re\)run\s+(?:Biber|BibTeX)",
        r"Rerun to get cross-references right",
        r"^!\s+.+",
        r"(?:LaTeX|Package\s+\S+)\s+Error:",
    )
    return [
        line.strip()
        for line in log_text.splitlines()
        if any(re.search(pattern, line, re.IGNORECASE) for pattern in patterns)
    ]


def check_artifacts(pdf_path, log_path, min_pages=None, max_pages=None):
    if not pdf_path.is_file() or not log_path.is_file():
        raise ValueError("The successful build must produce both a PDF and a log.")
    pages = page_count(pdf_path)
    log_text = log_path.read_text(errors="replace")
    issues = unresolved_messages(log_text)
    if min_pages is not None and pages < min_pages:
        issues.append(f"PDF has {pages} pages; minimum is {min_pages}.")
    if max_pages is not None and pages > max_pages:
        issues.append(f"PDF has {pages} pages; maximum is {max_pages}.")
    layout_warnings = [
        line.strip()
        for line in log_text.splitlines()
        if re.search(r"(?:Overfull|Underfull) \\[hv]box", line)
    ]
    return {
        "pdf": str(pdf_path),
        "log": str(log_path),
        "pages": pages,
        "page_bounds": [min_pages, max_pages],
        "issues": issues,
        "layout_warnings": layout_warnings,
        "visual_review_required": True,
        "scholarly_review_not_performed": True,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tex", type=Path, help="The actual final LaTeX root file")
    parser.add_argument("--output-dir", type=Path, help="Relative to the root directory")
    parser.add_argument("--min-pages", type=int, help="Optional minimum page count")
    parser.add_argument("--max-pages", type=int, help="Optional maximum page count")
    parser.add_argument("--engine", choices=("xelatex", "pdflatex", "lualatex"), default="xelatex")
    args = parser.parse_args()
    if (args.min_pages is not None and args.min_pages < 1) or (args.max_pages is not None and args.max_pages < 1):
        parser.error("Page bounds must be positive.")
    if args.min_pages is not None and args.max_pages is not None and args.max_pages < args.min_pages:
        parser.error("max-pages must be at least min-pages.")
    root = args.tex.resolve()
    if not root.is_file() or root.suffix.lower() != ".tex":
        parser.error("The root must be an existing .tex file.")
    missing = [name for name in ("latexmk", args.engine, "pdfinfo") if not shutil.which(name)]
    if missing:
        parser.error("Missing required tools: " + ", ".join(missing))
    output = args.output_dir or Path("build")
    if not output.is_absolute():
        output = root.parent / output
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    command = [
        "latexmk", {"xelatex": "-xelatex", "pdflatex": "-pdf", "lualatex": "-lualatex"}[args.engine], "-interaction=nonstopmode", "-halt-on-error",
        "-file-line-error", "-no-shell-escape", f"-outdir={output}", root.name,
    ]
    result = subprocess.run(
        command, cwd=root.parent, capture_output=True, text=True
    )
    if result.returncode:
        print("LaTeX build failed; no verified PDF was produced.", file=sys.stderr)
        print((result.stdout + result.stderr)[-6000:], file=sys.stderr)
        return 1
    try:
        report = check_artifacts(
            output / (root.stem + ".pdf"), output / (root.stem + ".log"),
            args.min_pages, args.max_pages,
        )
    except (ValueError, OSError, subprocess.CalledProcessError) as error:
        print(f"Verification failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if report["issues"]:
        return 1
    print("Build and reference checks passed. Inspect the output and assess scholarly content separately.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
