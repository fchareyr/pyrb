import logging

import numba
import numpy as np

from . import tools
from .settings import ADMM_TOL, CCD_CONVERGENCE_TOL, MAX_ITER, MAX_WEIGHT


@numba.njit
def accelerate(_varphi, r, s, u, alpha=10, tau=2):
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


@numba.jit(
    "Tuple((float64[:], float64[:], float64))(float64[:], float64, float64[:], float64, float64, float64[:], float64[:], float64[:], float64[:,:], float64, float64[:,:])",
    nopython=True,
)
def _cycle(x, c, var, _varphi, sigma_x, Sx, budgets, expected_returns, bounds, lambda_log, cov):
    """
    Internal numba function for computing one cycle of the CCD algorithm.

    """
    n = len(x)
    for i in range(n):
        alpha = c * var[i] + _varphi * sigma_x
        beta = c * (Sx[i] - x[i] * var[i]) - expected_returns[i] * sigma_x
        gamma_ = -lambda_log * budgets[i] * sigma_x
        x_tilde = (-beta + np.sqrt(beta**2 - 4 * alpha * gamma_)) / (2 * alpha)

        x_tilde = np.maximum(np.minimum(x_tilde, bounds[i, 1]), bounds[i, 0])

        x[i] = x_tilde
        Sx = np.dot(np.ascontiguousarray(cov), np.ascontiguousarray(x))
        sigma_x = np.sqrt(np.dot(np.ascontiguousarray(Sx), np.ascontiguousarray(x)))
    return x, Sx, sigma_x


def solve_rb_ccd(
    cov, budgets=None, expected_returns=None, risk_aversion=1.0, bounds=None, lambda_log=1.0, _varphi=0.0
):
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
        lambda_log: Log penalty parameter.
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
        bounds = np.array(bounds * 1.0)

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

    budgets = tools.to_array(budgets)
    expected_returns = tools.to_array(expected_returns)
    var = np.array(np.diag(cov))
    sx = tools.to_array(np.dot(cov, x))
    sigma_x = np.sqrt(np.dot(sx, x))

    cvg = False
    iters = 0

    while not cvg:
        x, sx, sigma_x = _cycle(
            x, risk_aversion, var, _varphi, sigma_x, sx, budgets, expected_returns, bounds, lambda_log, cov
        )
        cvg = np.sum(np.array(x - x0) ** 2) <= CCD_CONVERGENCE_TOL
        x0 = x.copy()
        iters = iters + 1
        if iters >= MAX_ITER:
            logging.info(
                f"Maximum iteration reached during the CCD descent: {MAX_ITER}"
            )
            break

    return tools.to_array(x)


def solve_rb_admm_qp(
    cov,
    budgets=None,
    expected_returns=None,
    risk_aversion=None,
    C=None,
    d=None,
    bounds=None,
    lambda_log=1,
    _varphi=1,
):
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

    lambda_log : float
        Log penalty parameter.

    _varphi : float
        This parameters is only useful for solving ADMM-CCD algorithm should be zeros otherwise.

    Returns
    -------
    x : array shape(n,)
        The array of optimal solution.

    """

    def proximal_log(a, b, risk_aversion, budgets):
        delta = b * b - 4 * a * risk_aversion * budgets
        x = (b + np.sqrt(delta)) / (2 * a)
        return x

    cov = np.array(cov)
    n = np.shape(cov)[0]

    if bounds is None:
        bounds = np.zeros((n, 2))
        bounds[:, 1] = MAX_WEIGHT
    else:
        bounds = np.array(bounds * 1.0)

    if budgets is None:
        budgets = np.array([1.0 / n] * n)

    x0 = 1 / np.diag(cov) / (np.sum(1 / np.diag(cov)))

    x = x0 / 100
    z = x.copy()
    zprev = z
    u = np.zeros(len(x))
    cvg = False
    iters = 0
    expected_returns_vec = tools.to_array(expected_returns)
    identity_matrix = np.identity(n)

    while not cvg:
        # x-update
        x = tools.quadprog_solve_qp(
            cov + _varphi * identity_matrix,
            risk_aversion * expected_returns_vec + _varphi * (z - u),
            G=C,
            h=d,
            bounds=bounds,
        )

        # z-update
        z = proximal_log(_varphi, (x + u) * _varphi, -lambda_log, budgets)

        # u-update
        r = x - z
        s = _varphi * (z - zprev)
        u += x - z

        # convergence check
        cvg1 = sum((x - x0) ** 2)
        cvg2 = sum((x - z) ** 2)
        cvg3 = sum((z - zprev) ** 2)
        cvg = np.max([cvg1, cvg2, cvg3]) <= ADMM_TOL
        x0 = x.copy()
        zprev = z

        iters = iters + 1
        if iters >= MAX_ITER:
            logging.info(f"Maximum iteration reached: {MAX_ITER}")
            break

        # parameters update
        _varphi, u = accelerate(_varphi, r, s, u)

    return tools.to_array(x)


def solve_rb_admm_ccd(
    cov,
    budgets=None,
    expected_returns=None,
    risk_aversion=None,
    C=None,
    d=None,
    bounds=None,
    lambda_log=1,
    _varphi=1,
):
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

    lambda_log : float
        Log penalty parameter.

    _varphi : float
        This parameters is only useful for solving ADMM-CCD algorithm should be zeros otherwise.

    Returns
    -------
    x : array shape(n,)
        The array of optimal solution.


    """

    cov = np.array(cov)

    x0 = 1 / np.diag(cov) / (np.sum(1 / np.diag(cov)))

    x = x0 / 100
    z = x.copy()
    zprev = z
    u = np.zeros(len(x))
    cvg = False
    iters = 0
    expected_returns_vec = tools.to_array(expected_returns)
    while not cvg:
        # x-update
        x = solve_rb_ccd(
            cov,
            budgets=budgets,
            expected_returns=expected_returns_vec + (_varphi * (z - u)),
            bounds=bounds,
            lambda_log=lambda_log,
            risk_aversion=risk_aversion,
            _varphi=_varphi,
        )

        # z-update
        z = tools.proximal_polyhedra(x + u, C, d, A=None, b=None, bound=bounds)

        # u-update
        r = x - z
        s = _varphi * (z - zprev)
        u += x - z

        # convergence check
        cvg1 = sum((x - x0) ** 2)
        cvg2 = sum((x - z) ** 2)
        cvg3 = sum((z - zprev) ** 2)
        cvg = np.max([cvg1, cvg2, cvg3]) <= ADMM_TOL
        x0 = x.copy()
        zprev = z

        iters = iters + 1
        if iters >= MAX_ITER:
            logging.info(f"Maximum iteration reached: {MAX_ITER}")
            break

        # parameters update
        _varphi, u = accelerate(_varphi, r, s, u)

    return tools.to_array(x)
