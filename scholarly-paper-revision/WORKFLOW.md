# Writing together

You can write rough LaTeX, fragments, questions, and calculations in
`thesis/notes/inbox.tex`. Dates and short topic headings help, but polished prose
is unnecessary. Keep uncertain steps visibly uncertain. Source links, citation
keys, or a PDF filename and page number are enough to begin source checking.

Ask: “Turn the notes on [topic] into a thesis section; check the argument and
references, then build.” I will read the surrounding manuscript, work through
the calculation, develop the explanation, maintain notation, verify sources,
update the bibliography, and compile. Your scratch work stays intact. You review
the resulting argument and any research decisions that need your judgment.

## What is relevant in this package

- `SKILL.md`: the editorial and mathematical review policy. Preserve meaning,
  state assumptions, remove filler, and distinguish evidence from conjecture.
- `references/teaching-and-review.md`: useful guidance for motivation,
  definitions, worked derivations, and limits. Its example PDF is an external
  style reference, not a thesis source or prescribed chapter structure.
- `references/latex-workflow.md`: compile the real root, resolve citations and
  cross references, and inspect the resulting PDF.
- `assets/newcommands.tex`: optional notation library with finance/statistics
  macros, styled floats, and authorship decorations. It also contains suspicious
  definitions (for example `\tildebalpha` expands to a hat). Adopt only reviewed
  definitions needed by the thesis; do not import the whole file by default.
- `assets/survey-template.tex`: an article/survey starter, not a university
  thesis template. Its headings do not determine your thesis organization.
- `scripts/build_survey.py`: optional latexmk/Poppler verification helper.
  Those dependencies are currently unavailable; the starter below uses the
  installed pdfLaTeX/BibTeX tools instead.

## Workspace

`thesis/main.tex` is a provisional report root, pending a university class.
The seven files in `thesis/chapters/` follow the research specification;
`thesis/notes/inbox.tex` holds your notes and compiles separately through
`thesis/scratch.tex`. Both share `thesis/preamble.tex`.
`thesis/references.bib` holds verified bibliography entries.
`thesis/notes/editorial.md` records open issues and source checks.

From this folder:

```sh
python3 scripts/build_thesis.py
python3 scripts/build_thesis.py scratch
```

Outputs go to `thesis/build/main.pdf` and `thesis/build/scratch.pdf`.
The scratch build accepts either a fragment or a pasted standalone document in
the inbox. For standalone documents it previews the document body with the
shared thesis preamble; the original preamble and title metadata remain intact
in the inbox but are not loaded. Add any needed packages/macros to the shared
preamble after review. Use the Python command to generate the preview body.
The script uses pdfLaTeX, runs BibTeX when citations are present, repeats LaTeX
for references, and rejects unresolved references and errors in the final log.
Layout warnings are reported for review. Shell escape is disabled.
For a supplied template, adapt the engine and bibliography backend to its
requirements rather than converting the thesis to this provisional setup.

## Editorial agreement

Keep explanations at the level needed for the thesis audience. Introduce the
problem before machinery; define symbols and assumptions before use; retain
enough intermediate steps for you to learn from the argument. Maintain your
voice and avoid padding, repeated summaries, or invented significance.

Check a cited claim against the relevant original source, including conditions
and page/theorem locations when useful. Verify bibliographic metadata before
adding an entry. Record inaccessible sources as pending; never imply they were
read. Research unresolved citations using accessible primary sources when asked
to process a section. Do not fabricate a citation just to make the build pass.

Track each substantive unresolved issue in `notes/editorial.md`, with its
location, needed evidence or decision, and status. A session ends with a concise
report of changes, mathematical or source issues, and build/PDF checks. Routine
formatting needs no approval. Changes to the research question, interpretation,
or unsupported claims need your input. You retain final academic judgment and
should follow your institution's authorship and disclosure requirements.

The thesis topic, research question, proposed structure, and source locations
are now recorded in `THESIS_CONTEXT.md`, based on the local research documents.
University formatting, final citation style, and supervisor requirements remain
unspecified. The chapter skeleton follows the focused factor-risk specification;
it does not present planned experiments as completed research.
