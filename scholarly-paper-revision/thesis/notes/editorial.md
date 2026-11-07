# Editorial record

## Open

- Setup: university template, final bibliography style, and supervisor
  requirements remain unidentified. Topic and proposed structure are resolved.
- Probabilistic PCA and spiked covariance appear only as empty headings in the
  inbox. They remain future writing tasks; no substantive content was invented.

## Resolved

- Created a provisional manuscript and separately compilable notes workspace.
- Retained the supplied skill and assets without importing the broad macro file.
- Read the local focused specification, library index/review notes, broader DOCX
  plan and covariance background notes. Recorded their roles in THESIS_CONTEXT.md.
- Adopted the focused specification's working title and seven-chapter skeleton.
- Both roots compiled and final-log reference checks passed during initial setup;
  PDF appearance was not visually inspected. No source-paper verification is
  implied by reading the existing literature notes.

## PCA notes organised — 2026-11-07

- Source: `inbox.tex`, preserved unchanged, including its standalone preamble.
  Draft: `chapters/02-foundations.tex`, Section 2.1.
- Standardised asset/factor dimensions from n/m to p/k to match the project
  specification; retained the user's assets-by-time orientation p by T.
- Made sample centering explicit. Adopted 1/(T-1) sample covariance consistently
  with the project specification, replacing the inbox's 1/T convention and
  explaining the scaling. Distinguished sample S from population Sigma_r.
- Corrected the factor-return space to real matrices and the transformed factor
  notation to F-tilde = C^{-1} F. Separated invertible coordinate changes from
  transformations preserving orthonormal loadings; recorded boundary ties.
- Used compact SVD with explicit dimensions; supplied a projection/trace proof
  of truncated-SVD optimality and the sample factor/residual decomposition.
- Clarified that PCA's unit-norm exposure has no full-investment or long-only
  restriction. Sample identities do not establish population factor recovery.
- Added `eckartYoung1936`. Author names, title, year, journal, volume/issue,
  pages and DOI checked against the publisher record:
  https://www.cambridge.org/core/journals/psychometrika/article/abs/approximation-of-one-matrix-by-another-of-lower-rank/B29672E1EDD0FA1B7611D4DFAFC321B3
  The full historical paper was not accessed; the manuscript supplies its own
  finite-dimensional proof rather than claiming a full-text source audit.
- The standalone inbox refers to an unavailable style file. The scratch builder
  now extracts its document body into an ignored build artifact and previews it
  with the shared preamble, without changing the original notes. Its original
  title metadata and unused package list are not imported.
- Verification: main and scratch builds passed final-log reference checks.
  Final main log has no overfull/underfull boxes. Extracted PDF text and visually
  inspected all four PCA pages, including displayed equations and page breaks.
