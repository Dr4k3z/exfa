# ExFa

ExFa (Extended Factors) is a Python framework for fundamental and statistical
factor risk models. Both models share a NumPy-based fitted API for returns,
loadings, factor returns, residuals, and covariance estimates. Factor construction,
data providers, and diagnostic calculations are separate concerns.

## Code conventions

- Use clear, descriptive names. Functions and methods should start with a verb
  describing their action, such as `compute_loadings` or `validate_returns`.
  Properties describe values and use nouns, such as `loadings` or `covariance`.
- Use `loadings` consistently rather than `exposures`.
- Keep public APIs and array orientations consistent across models: returns and
  residuals are `(assets, times)`, loadings are `(assets, factors)`, and factor
  returns are `(factors, times)`.
- Keep model estimation NumPy-based. Perform pandas data preparation separately.
- Use type annotations and preserve precise types through decorators and descriptors.
- Use dataclasses only when they suit a simple record of related data. Do not use
  them by default or introduce unnecessary wrappers and abstractions.
- Put opening and closing docstring quotes on separate lines, with the text
  between them. Leave exactly one blank line between a docstring and following
  code. Do not use inline or single-line docstrings.
- Use pytest for tests. Cover numerical behavior, validation, and public contracts.
- Ask for permission before writing code or changing existing file contents unless
  the user has already approved those changes. Moving files alone does not require
  content-edit approval.

## Deployment and commit conventions

- At the end of every code-writing task, run Ruff and ty over the whole repository,
  not just changed files:

  ```sh
  uv run ruff check .
  uv run ty check .
  ```

- Before committing, also run the test suite:

  ```sh
  uv run pytest
  ```

- Resolve failures caused by the changes. Report any remaining failures and
  blockers accurately; do not claim checks passed when they did not.
- Keep commits focused, use descriptive action-oriented messages, and preserve
  unrelated user changes.
- Sign every agent-authored commit with a `Co-authored-by:` trailer identifying
  the agent and the actual model used. This attribution does not imply a
  cryptographic signature; do not claim GPG or SSH signing unless verified.
- PR descriptions should explain the problem, resulting behavior, compatibility
  changes, and validation. End every agent-authored PR description with a
  `Co-authored by` attribution identifying the agent and the actual model used.
- Commit, push, or deploy only within the scope authorized by the user.
