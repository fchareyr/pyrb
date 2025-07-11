# Copilot Coding Agent Instructions for pyrb

## Project Overview
- **Purpose:** Implements constrained and unconstrained risk budgeting portfolio allocation algorithms in Python, supporting advanced risk measures and constraints.
- **Core Domain:** Quantitative finance, portfolio optimization, risk parity, and risk budgeting.
- **Reference:** See [Richard & Roncalli (2019)](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3331184) for theoretical background.

## Architecture & Key Components
- **Source code:** All main logic is in `src/pyrb/`:
  - `allocation.py`: Main algorithms and classes (`EqualRiskContribution`, `RiskBudgeting`, `RiskBudgetingWithER`, `ConstrainedRiskBudgeting`).
  - `solvers.py`: Numerical solvers for optimization (ADMM, CCD, QP).
  - `tools.py`, `validation.py`: Utility and input validation functions.
  - `settings.py`: Global constants for optimization routines.
- **Tests:** Located in `tests/`, using `pytest`. Example: `test_risk_budgeting.py` covers all major allocation classes.
- **Notebooks:** `notebooks/` contains example workflows and data for interactive exploration.

## Developer Workflow
- **Install dependencies:**
  - **Always use `uv` for dependency management. Never use `pip`, `conda`, or `poetry`.**
  - Install: `uv add git+https://github.com/jcrichard/pyrb`
- **Run tests:** `pytest`
- **Lint/format:** `ruff format` and `ruff check --fix`
- **CI:** GitHub Actions runs tests and linting on push/PR; Dependabot updates dependencies.

## Project-Specific Patterns
- **Class-based API:** All allocation methods are implemented as classes with a common interface (`solve`, `get_risk_contributions`, etc.).
- **Covariance matrix:** All algorithms expect a NumPy array for covariance; input validation is strict.
- **Constraints:** Constrained problems use bounds and linear inequalities (`C`, `d`, `bounds`).
- **Risk budgets:** Passed as arrays; if `None`, defaults to equal risk.
- **Expected returns:** Optional, passed as `pi` arrays.
- **Solvers:** Selectable via string argument (`solver="admm_ccd"`, `"admm_qp"`, etc.).
- **Error handling:** Uses exceptions and logging for failed optimizations (see `ConstrainedRiskBudgeting.solve`).

## Integration Points
- **External dependencies:**
  - `numpy`, `scipy` for math/optimization
  - `ruff` for linting/formatting
  - `pytest` for testing
- **No external service calls or APIs.**

## Example Usage
```python
from pyrb import EqualRiskContribution
ERC = EqualRiskContribution(cov)
ERC.solve()
ERC.get_risk_contributions()
ERC.get_volatility()
```

## Key Files
- `src/pyrb/allocation.py`: Main entry point for algorithms
- `tests/test_risk_budgeting.py`: Canonical test cases
- `README.md`: Quickstart, install, and workflow reference

---
**If unclear or missing context, review `README.md`, `src/pyrb/allocation.py`, and test files for canonical patterns.**
