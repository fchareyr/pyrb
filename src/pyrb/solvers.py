import logging
from typing import Any

import numba
import numpy as np

from . import tools
from .settings import ADMM_TOL, CCD_CONVERGENCE_TOL, MAX_ITER, MAX_WEIGHT


@numba.njit("Tuple((float64, float64[:]))(float64, float64[:], float64[:], float64[:], float64, float64)")
def accelerate(
    _varphi: float,
    r: np.ndarray[Any, Any],
    s: np.ndarray[Any, Any],
    u: np.ndarray[Any, Any],
    alpha: float = 10,
    tau: float = 2,
) -> tuple[float, np.ndarray[Any, Any]]:
    """Update varphy and dual error for accelerating convergence after ADMM steps.

    Args:
        _varphi: Current varphi value.
        r: Primal error.
        s: Dual error.
        u: Primal error.
        alpha: Error threshold (default 10).
        tau: Scaling parameter (default 2).

    Returns:
        tuple: Updated varphi and primal_error.
    """

    primal_error = np.sum(r**2)
    dual_error = np.sum(s * s)
    if primal_error > alpha * dual_error:
        _varphi = _varphi * tau
        u = u / tau
    elif dual_error > alpha * primal_error:
        _varphi = _varphi / tau
        u = u * tau
    return _varphi, u


@numba.jit("Tuple((float64[:], float64[:], float64))(float64[:], float64, float64[:], float64, float64, float64[:], float64[:], float64[:], float64[:,:], float64, float64[:,:])", nopython=True)
def _cycle(
    x: np.ndarray[Any, Any],
    c: float,
    var: np.ndarray[Any, Any],
    _varphi: float,
    sigma_x: float,
    sx: np.ndarray[Any, Any],
    budgets: np.ndarray[Any, Any],
    expected_returns: np.ndarray[Any, Any],
    bounds: np.ndarray[Any, Any],
    llambda_log: float,
    cov: np.ndarray[Any, Any],
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any], float]:
    """
    Internal numba function for computing one cycle of the CCD algorithm.

    """
    n = len(x)
    for i in range(n):
        alpha = c * var[i] + _varphi * sigma_x
        beta = c * (sx[i] - x[i] * var[i]) - expected_returns[i] * sigma_x
        gamma_ = -llambda_log * budgets[i] * sigma_x
        x_tilde = (-beta + np.sqrt(beta**2 - 4 * alpha * gamma_)) / (2 * alpha)

        x_tilde = np.maximum(np.minimum(x_tilde, bounds[i, 1]), bounds[i, 0])

        x[i] = x_tilde
        sx = np.dot(np.ascontiguousarray(cov), np.ascontiguousarray(x))
        sigma_x = np.sqrt(np.dot(np.ascontiguousarray(sx), np.ascontiguousarray(x)))
    return x, sx, sigma_x


def solve_rb_ccd(
    cov: np.ndarray[Any, Any],
    budgets: np.ndarray[Any, Any] | None = None,
    expected_returns: np.ndarray[Any, Any] | None = None,
    risk_aversion: float = 1.0,
    bounds: np.ndarray[Any, Any] | None = None,
    llambda_log: float = 1.0,
    _varphi: float = 0.0,
) -> np.ndarray[Any, Any]:
    """Solve the risk budgeting problem using cyclical coordinate descent.

    Solves the risk budgeting problem for standard deviation risk-based measure with
    bounds constraints using cyclical coordinate descent (CCD). It corresponds to
    solving equation (17) in the paper.

    By default the function solves the ERC portfolio or the RB portfolio if budgets are given.

    Args:
        cov: Covariance matrix of the returns, shape (n, n).
        budgets: Risk budgets for each asset, shape (n,).
            Default is None which implies equal risk budget.
        expected_returns: Expected excess return for each asset, shape (n,).
            Default is None which implies 0 for each asset.
        risk_aversion: Risk aversion parameter, default is 1.
        bounds: Array of minimum and maximum bounds, shape (n, 2).
            If None the default bounds are [0,1].
        llambda_log: Log penalty parameter.
        _varphi: This parameter is only useful for solving ADMM-CCD algorithm,
            should be zeros otherwise.

    Returns:
        Optimal solution array, shape (n,).
    """

    n = cov.shape[0]

    if bounds is None:
        bounds = np.zeros((n, 2))
        bounds[:, 1] = MAX_WEIGHT
    else:
        bounds = np.array(bounds, dtype=float)

    if budgets is None:
        budgets = np.array([1.0] * n) / n
    else:
        budgets = np.array(budgets)
    budgets = budgets / np.sum(budgets)

    if (risk_aversion is None) | (expected_returns is None):
        risk_aversion = 1.0
        expected_returns = np.array([0.0] * n)
    else:
        risk_aversion = float(risk_aversion)
        expected_returns = np.array(expected_returns).astype(float)

    # initial value equals to 1/vol portfolio
    x = 1 / np.diag(cov) ** 0.5 / (np.sum(1 / np.diag(cov) ** 0.5))
    x0 = x / 100

    budgets_arr = tools.to_array(budgets)
    expected_returns_arr = tools.to_array(expected_returns)
    var = np.array(np.diag(cov))
    sx = np.dot(cov, x)
    sigma_x = np.sqrt(np.dot(sx, x))
    if budgets_arr is None:
        raise ValueError("budgets cannot be None after conversion")
    if expected_returns_arr is None:
        raise ValueError("expected_returns cannot be None after conversion")

    cvg = False
    iters = 0

    while not cvg:
        x, sx, sigma_x = _cycle(
            x, risk_aversion, var, _varphi, sigma_x, sx, budgets_arr, expected_returns_arr, bounds, llambda_log, cov
        )
        cvg = np.sum(np.array(x - x0) ** 2) <= CCD_CONVERGENCE_TOL
        x0 = x.copy()
        iters = iters + 1
        if iters >= MAX_ITER:
            logging.info(
                f"Maximum iteration reached during the CCD descent: {MAX_ITER}"
            )
            break

    return x


def solve_rb_admm_qp(
    cov: np.ndarray[Any, Any],
    budgets: np.ndarray[Any, Any] | None = None,
    expected_returns: np.ndarray[Any, Any] | None = None,
    risk_aversion: float = 1.0,
    c: np.ndarray[Any, Any] | None = None,
    d: np.ndarray[Any, Any] | None = None,
    bounds: np.ndarray[Any, Any] | None = None,
    llambda_log: float = 1.0,
    _varphi: float = 1.0,
) -> np.ndarray[Any, Any]:
    """
    Solve the constrained risk budgeting constraint for the Mean Variance risk measure:
    The risk measure is given by R(x) =  x^T cov x - c * expected_returns^T x

    Parameters
    ----------
    cov : array, shape (n, n)
        Covariance matrix of the returns.

    budgets : array, shape (n,)
        Risk budgets for each asset (the default is None which implies equal risk budget).

    expected_returns : array, shape (n,)
        Expected excess return for each asset (the default is None which implies 0 for each asset).

    risk_aversion : float
        Risk aversion parameter equals to one by default.

    C : array, shape (p, n)
        Array of p inequality constraints. If None the problem is unconstrained and solved using CCD
        (algorithm 3) and it solves equation (17).

    d : array, shape (p,)
        Array of p constraints that matches the inequalities.

    bounds : array, shape (n, 2)
        Array of minimum and maximum bounds. If None the default bounds are [0,1].

    llambda_log : float
        Log penalty parameter.

    _varphi : float
        This parameters is only useful for solving ADMM-CCD algorithm should be zeros otherwise.

    Returns
    -------
    x : array shape(n,)
        The array of optimal solution.

    """

def proximal_log(a: float, b: np.ndarray[Any, Any], c: float, budgets: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    delta = b * b - 4 * a * c * budgets
    x = (b + np.sqrt(delta)) / (2 * a)
    return x

    cov = np.array(cov)
    n = np.shape(cov)[0]

    if bounds is None:
        bounds = np.zeros((n, 2))
        bounds[:, 1] = MAX_WEIGHT
    else:
        bounds = np.array(bounds, dtype=float)

    if budgets is None:
        budgets = np.array([1.0 / n] * n)

    x0 = 1 / np.diag(cov) / (np.sum(1 / np.diag(cov)))
    x = x0 / 100
    z = x.copy()
    zprev = z.copy()
    u = np.zeros(len(x))
    cvg = False
    iters = 0
    expected_returns_vec = tools.to_array(expected_returns)
    if expected_returns_vec is None:
        expected_returns_vec = np.zeros(n)
    identity_matrix = np.identity(n)
    while not cvg:
        # x-update
        x = tools.quadprog_solve_qp(
            cov + _varphi * identity_matrix,
            risk_aversion * expected_returns_vec + _varphi * (z - u),
            g=c,
            h=d,
            bounds=bounds,
        )
        # z-update
        z = proximal_log(_varphi, (x + u) * _varphi, -llambda_log, budgets)
        # u-update
        r = x - z
        s = _varphi * (z - zprev)
        u += x - z
        # convergence check
        cvg1 = np.sum((x - x0) ** 2)
        cvg2 = np.sum((x - z) ** 2)
        cvg3 = np.sum((z - zprev) ** 2)
        cvg = np.max([cvg1, cvg2, cvg3]) <= ADMM_TOL
        x0 = x.copy()
        zprev = z.copy()
        iters += 1
        if iters >= MAX_ITER:
            logging.info(f"Maximum iteration reached: {MAX_ITER}")
            break
        # parameters update
        _varphi, u = accelerate(_varphi, r, s, u)
    return x


def solve_rb_admm_ccd(
    cov: np.ndarray[Any, Any],
    budgets: np.ndarray[Any, Any] | None = None,
    expected_returns: np.ndarray[Any, Any] | None = None,
    risk_aversion: float = 1.0,
    c: np.ndarray[Any, Any] | None = None,
    d: np.ndarray[Any, Any] | None = None,
    bounds: np.ndarray[Any, Any] | None = None,
    llambda_log: float = 1.0,
    _varphi: float = 1.0,
) -> np.ndarray[Any, Any]:
    """
    Solve the constrained risk budgeting constraint for the standard deviation risk measure:
    The risk measure is given by R(x) = c * sqrt(x^T cov x) -  expected_returns^T x

    Parameters
    ----------
    Parameters
    ----------
    cov : array, shape (n, n)
        Covariance matrix of the returns.

    budgets : array, shape (n,)
        Risk budgets for each asset (the default is None which implies equal risk budget).

    expected_returns : array, shape (n,)
        Expected excess return for each asset (the default is None which implies 0 for each asset).

    risk_aversion : float
        Risk aversion parameter equals to one by default.

    C : array, shape (p, n)
        Array of p inequality constraints. If None the problem is unconstrained and solved using CCD
        (algorithm 3) and it solves equation (17).

    d : array, shape (p,)
        Array of p constraints that matches the inequalities.

    bounds : array, shape (n, 2)
        Array of minimum and maximum bounds. If None the default bounds are [0,1].

    llambda_log : float
        Log penalty parameter.

    _varphi : float
        This parameters is only useful for solving ADMM-CCD algorithm should be zeros otherwise.

    Returns
    -------
    x : array shape(n,)
        The array of optimal solution.


    """

    cov = np.array(cov)
    n = cov.shape[0]
    if bounds is None:
        bounds = np.zeros((n, 2))
        bounds[:, 1] = MAX_WEIGHT
    else:
        bounds = np.array(bounds, dtype=float)
    if budgets is None:
        budgets = np.array([1.0 / n] * n)
    x0 = 1 / np.diag(cov) / (np.sum(1 / np.diag(cov)))
    x = x0 / 100
    z = x.copy()
    zprev = z.copy()
    u = np.zeros(len(x))
    cvg = False
    iters = 0
    expected_returns_vec = tools.to_array(expected_returns)
    if expected_returns_vec is None:
        expected_returns_vec = np.zeros_like(x)
    # Ensure c and d are not None
    c_arr = c if c is not None else np.zeros((1, n))
    d_arr = d if d is not None else np.zeros(1)
    while not cvg:
        # x-update
        x = tools.proximal_polyhedra(x + u, c_arr, d_arr, a=None, b=None, bound=bounds)
        # z-update
        z = tools.proximal_polyhedra(x + u, c_arr, d_arr, a=None, b=None, bound=bounds)
        # u-update
        u += x - z
        # convergence check
        cvg1 = np.sum((x - x0) ** 2)
        cvg2 = np.sum((x - z) ** 2)
        cvg3 = np.sum((z - zprev) ** 2)
        cvg = np.max([cvg1, cvg2, cvg3]) <= ADMM_TOL
        x0 = x.copy()
        zprev = z.copy()
        iters += 1
        if iters >= MAX_ITER:
            logging.info(f"Maximum iteration reached: {MAX_ITER}")
            break
        # parameters update
        _varphi, u = accelerate(_varphi, x - z, z - zprev, u)
    return x
