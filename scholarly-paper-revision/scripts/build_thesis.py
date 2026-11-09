#!/usr/bin/env python3
"""Build the provisional manuscript or scratch notes with installed TeX tools."""
import argparse
from pathlib import Path
import re
import shutil
import subprocess
import sys

from build_survey import unresolved_messages


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("target", nargs="?", choices=("main", "scratch"), default="main")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1] / "thesis"
    output = root / "build"
    output.mkdir(exist_ok=True)
    if not shutil.which("pdflatex"):
        parser.error("pdflatex is required")

    def run(command, cwd):
        result = subprocess.run(command, cwd=cwd, capture_output=True, text=True)
        if result.returncode:
            print((result.stdout + result.stderr)[-6000:], file=sys.stderr)
            raise RuntimeError("Build failed; any previous PDF is not verified.")

    command = ["pdflatex", "-interaction=nonstopmode", "-halt-on-error",
               "-file-line-error", "-no-shell-escape", "-output-directory=build",
               args.target + ".tex"]
    try:
        if args.target == "scratch":
            notes = (root / "notes/inbox.tex").read_text()
            if "\\documentclass" in notes:
                start = "\\begin{document}"
                end = "\\end{document}"
                if start not in notes or end not in notes:
                    raise RuntimeError("Standalone inbox needs document delimiters.")
                notes = notes.split(start, 1)[1].split(end, 1)[0]
                print("Previewing inbox body with the shared preamble; original preamble is preserved but not loaded.")
            (output / "inbox-body.tex").write_text(notes)
        run(command, root)
        aux = (output / (args.target + ".aux")).read_text()
        if "\\citation{" in aux:
            if "\\bibdata{" not in aux or "\\bibstyle{" not in aux:
                raise RuntimeError("Citations need bibliography and bibliographystyle declarations.")
            if not shutil.which("bibtex"):
                raise RuntimeError("bibtex is required for this bibliography.")
            # Run from the source directory so references.bib resolves naturally.
            run(["bibtex", "build/" + args.target], root)
        for _ in range(2):
            run(command, root)
        log = (output / (args.target + ".log")).read_text(errors="replace")
        issues = unresolved_messages(log)
        warnings = [line for line in log.splitlines()
                    if re.search(r"(?:Overfull|Underfull) \\[hv]box", line)]
        for line in issues + warnings:
            print(line)
        if issues:
            raise RuntimeError("Final log contains unresolved errors or references.")
        print(f"Build and reference checks passed: {output / (args.target + '.pdf')}")
        print("PDF appearance and mathematical content require separate review.")
        return 0
    except (OSError, RuntimeError) as error:
        print(error, file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
