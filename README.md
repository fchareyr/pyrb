Constrained and Unconstrained Risk Budgeting Allocation in Python
================

[![Actions Status](https://github.com/fchareyr/pyrb/workflows/Python%20application/badge.svg)](https://github.com/fchareyr/pyrb/actions)
[![Code style: ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)


This repository contains the code for solving constrained risk budgeting
with generalized standard deviation-based risk measure:

<a href="https://www.codecogs.com/eqnedit.php?latex=R(x)&space;=&space;-&space;\pi^T&space;x&space;&plus;&space;c&space;\sqrt{&space;x^T&space;\Sigma&space;x}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?R(x)&space;=&space;-&space;\pi^T&space;x&space;&plus;&space;c&space;\sqrt{&space;x^T&space;\Sigma&space;x}" title="R(x) = - \pi^T x + c \sqrt{ x^T \Sigma x}" /></a>


This formulation encompasses Gaussian value-at-risk and Gaussian expected shortfall and the volatility. The algorithm supports bounds constraints and inequality constraints. It is is efficient for large dimension and suitable for backtesting. 

A description can be found in [*Constrained Risk Budgeting Portfolios: Theory, Algorithms, Applications & Puzzles*](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3331184)
by Jean-Charles Richard and Thierry Roncalli.

You can solve
------------------

- Equally risk contribution
- Risk budgeting
- Risk parity with expected return
- Constrained Risk parity

Installation
------------------
 Can be done using ``pip``: 

    pip install git+https://github.com/jcrichard/pyrb

 Or using ``uv`` (recommended for Python 3.12+):

    uv add git+https://github.com/jcrichard/pyrb


Usage
------------------

    from pyrb import EqualRiskContribution

    ERC = EqualRiskContribution(cov)
    ERC.solve()
    ERC.get_risk_contributions()
    ERC.get_volatility()


Development
------------------

This project uses modern Python development tools:

### Code Quality
- **ruff**: For linting and formatting
- **pytest**: For testing

### Installation for Development
```bash
# Clone the repository
git clone https://github.com/fchareyr/pyrb.git
cd pyrb

# Install in development mode
pip install -e .[dev]
```

### Running Tests
```bash
pytest
```

### Code Formatting and Linting
```bash
# Format code
ruff format

# Lint code
ruff check

# Fix linting issues automatically
ruff check --fix
```

### CI/CD
- GitHub Actions workflow runs on every push and pull request
- Tests are run on Python 3.12 and 3.13
- Code must pass linting and formatting checks
- Dependabot automatically updates dependencies


References
------------------

>Griveau-Billion, T., Richard, J-C., and Roncalli, T. (2013), A Fast Algorithm for Computing High-dimensional Risk Parity Portfolios, SSRN.

>Maillard, S., Roncalli, T. and
    Teiletche, J. (2010), The Properties of Equally Weighted Risk Contribution Portfolios,
    Journal of Portfolio Management, 36(4), pp. 60-70.
    
>Richard, J-C., and Roncalli, T. (2015), Smart
    Beta: Managing Diversification of Minimum Variance Portfolios, in Jurczenko, E. (Ed.),
    Risk-based and Factor Investing, ISTE Press -- Elsevier.
    
>Richard, J-C., and Roncalli, T. (2019), Constrained Risk Budgeting Portfolios: Theory, Algorithms, Applications & Puzzles, SSRN.
    
>Roncalli, T. (2015), Introducing Expected Returns into Risk Parity Portfolios: A New Framework for Asset Allocation,
    Bankers, Markets & Investors, 138, pp. 18-28.
 
