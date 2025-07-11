import numpy as np

from .settings import RISK_BUDGET_TOL


def check_covariance(cov):
    """Check if the covariance matrix is valid.

    Args:
        cov: Covariance matrix to validate.

    Raises:
        ValueError: If matrix is not square or contains missing values.
    """
    if cov.shape[0] != cov.shape[1]:
        raise ValueError("The covariance matrix is not squared")
    if np.isnan(cov).sum().sum() > 0:
        raise ValueError("The covariance matrix contains missing values")


def check_expected_return(mu, n):
    """Check if the expected return vector is valid.

    Args:
        mu: Expected return vector to validate.
        n: Number of assets.

    Raises:
        ValueError: If vector size doesn't match number of assets or contains missing values.
    """
    if mu is None:
        return
    if n != len(mu):
        raise ValueError(
            "Expected returns vector size is not equal to the number of asset."
        )
    if np.isnan(mu).sum() > 0:
        raise ValueError("The expected returns vector contains missing values")


def check_constraints(C, d, n):
    """Check if the constraint matrix and vector are valid.

    Args:
        C: Constraint matrix.
        d: Constraint vector.
        n: Number of assets.

    Raises:
        ValueError: If matrix dimensions don't match or contain missing values.
    """
    if C is None:
        return
    if n != C.shape[1]:
        raise ValueError("Number of columns of C is not equal to the number of asset.")
    if len(d) != C.shape[0]:
        raise ValueError("Number of rows of C is not equal to the length  of d.")


def check_bounds(bounds, n):
    """Check if the bounds array is valid.

    Args:
        bounds: Bounds array to validate.
        n: Number of assets.

    Raises:
        ValueError: If bounds dimensions don't match or contain invalid values.
    """
    if bounds is None:
        return
    if n != bounds.shape[0]:
        raise ValueError(
            "The number of rows of the bounds array is not equal to the number of asset."
        )
    if 2 != bounds.shape[1]:
        raise ValueError(
            "The number of columns the bounds array should be equal to two (min and max bounds)."
        )


def check_risk_budget(riskbudgets, n):
    """Check if the risk budget vector is valid.

    Args:
        riskbudgets: Risk budget vector to validate.
        n: Number of assets.

    Raises:
        ValueError: If vector size doesn't match number of assets or contains invalid values.
    """
    if riskbudgets is None:
        return
    if np.isnan(riskbudgets).sum() > 0:
        raise ValueError("Risk budget contains missing values")
    if (np.array(riskbudgets) < 0).sum() > 0:
        raise ValueError("Risk budget contains negative values")
    if n != len(riskbudgets):
        raise ValueError("Risk budget size is not equal to the number of asset.")
    if all(v < RISK_BUDGET_TOL for v in riskbudgets):
        raise ValueError(
            f"One of the budget is smaller than {RISK_BUDGET_TOL}. If you want a risk budget of 0 please remove the asset."
        )
