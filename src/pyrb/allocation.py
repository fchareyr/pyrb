import logging
from abc import abstractmethod

import numpy as np
import scipy.optimize as optimize

from . import tools, validation
from .settings import BISECTION_UPPER_BOUND, MAX_ITER_BISECTION
from .solvers import solve_rb_admm_ccd, solve_rb_admm_qp, solve_rb_ccd


class RiskBudgetAllocation:
    @property
    def cov(self):
        return self.__cov

    @property
    def weights(self):
        return self._weights

    @property
    def expected_returns(self):
        return self.__expected_returns

    @property
    def n(self):
        return self.__n

    def __init__(self, cov, expected_returns=None, weights=None):
        """Base class for Risk Budgeting Allocation.

        Args:
            cov: Covariance matrix of the returns, shape (n, n).
            expected_returns: Expected excess return for each asset, shape (n,).
                The default is None which implies 0 for each asset.
            weights: Array of weights, shape (n,).
        """
        self.__n = cov.shape[0]
        if weights is None:
            weights = np.array([np.nan] * self.n)
        self._weights = weights
        validation.check_covariance(cov)

        if expected_returns is None:
            expected_returns = np.array([0.0] * self.n)
        validation.check_expected_return(expected_returns, self.n)
        self.__expected_returns = tools.to_column_matrix(expected_returns)

        self.__cov = np.array(cov)
        self.lambda_star = np.nan

    @abstractmethod
    def solve(self):
        """Solve the problem.

        This is an abstract method that must be implemented by subclasses.
        """
        pass

    @abstractmethod
    def get_risk_contributions(self):
        """Get the risk contribution of the Risk Budgeting Allocation."""
        pass

    def get_variance(self):
        """Get the portfolio variance: weights.T * cov * weights.

        Returns:
            Portfolio variance as a float.
        """
        weights = self.weights
        cov = self.cov
        weights = tools.to_column_matrix(weights)
        cov = np.asarray(cov)
        RC = np.multiply(weights, cov @ weights)
        return np.sum(tools.to_array(RC))

    def get_volatility(self):
        """Get the portfolio volatility: sqrt(x.T * cov * x).

        Returns:
            Portfolio volatility as a float.
        """
        return self.get_variance() ** 0.5

    def get_expected_return(self):
        """Get the portfolio expected excess returns: weights.T * expected_returns.

        Returns:
            Portfolio expected excess return as a float, or NaN if expected_returns is None.
        """
        if self.expected_returns is None:
            return np.nan
        else:
            weights = self.weights
            weights = tools.to_column_matrix(weights)
        return float(weights.T @ self.expected_returns)

    def __str__(self):
        return (
            f"solution x: {np.round(self.weights * 100, 4)}\n"
            f"lambda star: {np.round(self.lambda_star * 100, 4)}\n"
            f"risk contributions: {np.round(self.get_risk_contributions() * 100, 4)}\n"
            f"sigma(x): {np.round(self.get_volatility() * 100, 4)}\n"
            f"sum(x): {np.round(self.weights.sum() * 100, 4)}\n"
        )


class EqualRiskContribution(RiskBudgetAllocation):
    def __init__(self, cov):
        """Solve the equal risk contribution problem using cyclical coordinate descent.

        Although this does not change the optimal solution, the risk measure
        considered is the portfolio volatility.

        Args:
            cov: Covariance matrix of the returns, shape (n, n).
        """

        RiskBudgetAllocation.__init__(self, cov)

    def solve(self):
        """Solve the equal risk contribution problem using cyclical coordinate descent.

        Updates the internal weights and lambda_star attributes.
        """
        weights = solve_rb_ccd(cov=self.cov)
        self._weights = tools.to_array(weights / weights.sum())
        self.lambda_star = self.get_volatility()

    def get_risk_contributions(self, scale=True):
        weights = self.weights
        cov = self.cov
        weights = tools.to_column_matrix(weights)
        cov = np.asarray(cov)
        RC = np.multiply(weights, cov @ weights) / self.get_volatility()
        if scale:
            RC = RC / RC.sum()
        return tools.to_array(RC)


class RiskBudgeting(RiskBudgetAllocation):
    def __init__(self, cov, budgets):
        """Solve the risk budgeting problem using cyclical coordinate descent.

        Although this does not change the optimal solution, the risk measure
        considered is the portfolio volatility.

        Args:
            cov: Covariance matrix of the returns, shape (n, n).
            budgets: Risk budgets for each asset, shape (n,).
        """
        RiskBudgetAllocation.__init__(self, cov=cov)
        validation.check_risk_budget(budgets, self.n)
        self.budgets = budgets

    def solve(self):
        """Solve the risk budgeting problem using cyclical coordinate descent.

        Updates the internal weights (x) and lambda_star attributes.
        """
        weights = solve_rb_ccd(cov=self.cov, budgets=self.budgets)
        self._weights = tools.to_array(weights / weights.sum())
        self.lambda_star = self.get_volatility()

    def get_risk_contributions(self, scale=True):
        weights = self.weights
        cov = self.cov
        weights = tools.to_column_matrix(weights)
        cov = np.asarray(cov)
        RC = np.multiply(weights, cov @ weights) / self.get_volatility()
        if scale:
            RC = RC / RC.sum()
        return tools.to_array(RC)


class RiskBudgetingWithER(RiskBudgetAllocation):
    def __init__(self, cov, budgets=None, expected_returns=None, risk_aversion=1):
        """Solve the risk budgeting problem for the standard deviation risk measure.

        Uses cyclical coordinate descent. The risk measure is given by
        R(x) = c * sqrt(weights^T cov weights) - expected_returns^T weights.

        Args:
            cov: Covariance matrix of the returns, shape (n, n).
            budgets: Risk budgets for each asset, shape (n,).
                Default is None which implies equal risk budget.
            expected_returns: Expected excess return for each asset, shape (n,).
                Default is None which implies 0 for each asset.
            risk_aversion: Risk aversion parameter, default is 1.
        """
        RiskBudgetAllocation.__init__(self, cov=cov, expected_returns=expected_returns)
        validation.check_risk_budget(budgets, self.n)
        self.budgets = budgets
        self.risk_aversion = risk_aversion

    def solve(self):
        weights = solve_rb_ccd(
            cov=self.cov,
            budgets=self.budgets,
            expected_returns=self.expected_returns,
            risk_aversion=self.risk_aversion,
        )
        self._weights = tools.to_array(weights / weights.sum())
        self.lambda_star = (
            -self.get_expected_return() + self.get_volatility() * self.risk_aversion
        )

    def get_risk_contributions(self, scale=True):
        weights = self.weights
        cov = self.cov
        weights = tools.to_column_matrix(weights)
        cov = np.asarray(cov)
        RC = (
            np.multiply(weights, cov @ weights)
            / self.get_volatility()
            * self.risk_aversion
            - self.weights * self.expected_returns
        )
        if scale:
            RC = RC / RC.sum()
        return tools.to_array(RC)

    def __str__(self):
        return (
            super().__str__()
            + f"mu(x): {np.round(self.get_expected_return() * 100, 4)}\n"
        )


class ConstrainedRiskBudgeting(RiskBudgetingWithER):
    def __init__(
        self,
        cov,
        budgets=None,
        expected_returns=None,
        risk_aversion=1,
        C=None,
        d=None,
        bounds=None,
        solver="admm_ccd",
    ):
        """Solve the constrained risk budgeting problem.

        Supports linear inequality (Cx <= d) and bounds constraints.
        Notations follow the paper Constrained Risk Budgeting Portfolios
        by Richard J-C. and Roncalli T. (2019).

        Args:
            cov: Covariance matrix of the returns, shape (n, n).
            budgets: Risk budgets for each asset, shape (n,).
                Default is None which implies equal risk budget.
            expected_returns: Expected excess return for each asset, shape (n,).
                Default is None which implies 0 for each asset.
            risk_aversion: Risk aversion parameter, default is 1.
            C: Array of p inequality constraints, shape (p, n). If None the
                problem is unconstrained and solved using CCD (algorithm 3)
                and it solves equation (17).
            d: Array of p constraints that matches the inequalities, shape (p,).
            bounds: Array of minimum and maximum bounds, shape (n, 2).
                If None the default bounds are [0,1].
            solver: Solver method, either "admm_ccd" (default) or "admm_qp".
                "admm_ccd": generalized standard deviation-based risk measure +
                linear constraints. The algorithm is ADMM_CCD (algorithm 4) and
                it solves equation (14).
                "admm_qp": mean variance risk measure + linear constraints.
                The algorithm is ADMM_QP and it solves equation (15).
        """

        RiskBudgetingWithER.__init__(
            self,
            cov=cov,
            budgets=budgets,
            expected_returns=expected_returns,
            risk_aversion=risk_aversion,
        )

        self.d = d
        self.C = C
        self.bounds = bounds
        validation.check_bounds(bounds, self.n)
        validation.check_constraints(C, d, self.n)
        self.solver = solver
        if (self.solver == "admm_qp") and (self.expected_returns is not None):
            logging.warning(
                "The solver is set to 'admm_qp'. The risk measure is the mean variance in this case. The optimal "
                "solution will not be the same than 'admm_ccd' when expected_returns is not zero.     "
            )

    def __str__(self):
        if self.C is not None:
            return (
                f"solver: {self.solver}\n"
                + "----------------------------\n"
                + super().__str__()
                + f"C@x: {self.C @ self.weights}\n"
            )
        else:
            return super().__str__()

    def _sum_to_one_constraint(self, _lambda):
        weights = self._lambda_solve(_lambda)
        sum_weights = sum(weights)
        return sum_weights - 1

    def _lambda_solve(self, _lambda):
        if (
            self.C is None
        ):  # it is optimal to take the CCD in case of separable constraints
            weights = solve_rb_ccd(
                self.cov,
                self.budgets,
                self.expected_returns,
                self.risk_aversion,
                self.bounds,
                _lambda,
            )
            self.solver = "ccd"
        elif self.solver == "admm_qp":
            weights = solve_rb_admm_qp(
                cov=self.cov,
                budgets=self.budgets,
                expected_returns=self.expected_returns,
                risk_aversion=self.risk_aversion,
                C=self.C,
                d=self.d,
                bounds=self.bounds,
                lambda_log=_lambda,
            )
        elif self.solver == "admm_ccd":
            weights = solve_rb_admm_ccd(
                cov=self.cov,
                budgets=self.budgets,
                expected_returns=self.expected_returns,
                risk_aversion=self.risk_aversion,
                C=self.risk_aversion,
                d=self.d,
                bounds=self.bounds,
                lambda_log=_lambda,
            )
        return weights

    def solve(self):
        try:
            lambda_star = optimize.bisect(
                self._sum_to_one_constraint,
                0,
                BISECTION_UPPER_BOUND,
                maxiter=MAX_ITER_BISECTION,
            )
            self.lambda_star = lambda_star
            self._weights = self._lambda_solve(lambda_star)
        except Exception as e:
            if e.args[0] == "f(a) and f(b) must have different signs":
                logging.exception(
                    "Bisection failed: "
                    + str(e)
                    + ". If you are using expected returns the parameter 'c' need to be correctly scaled (see remark 1 in the paper). Otherwise please check the constraints or increase the bisection upper bound."
                )
            else:
                logging.exception("Problem not solved: " + str(e))

    def get_risk_contributions(self, scale=True):
        """Return the risk contribution.

        If the solver is "admm_qp" the mean variance risk measure is considered.

        Args:
            scale: If True, the sum on risk contribution is scaled to one.

        Returns:
            Risk contribution of each asset, shape (n,).
        """
        weights = self.weights
        cov = self.cov
        weights = tools.to_column_matrix(weights)
        cov = np.asarray(cov)

        if self.solver == "admm_qp":
            RC = (
                np.multiply(weights, cov @ weights)
                - self.risk_aversion * self.weights * self.expected_returns
            )
        else:
            RC = np.multiply(
                weights, cov @ weights
            ).T / self.get_volatility() * self.risk_aversion - tools.to_array(
                self.weights.T
            ) * tools.to_array(self.expected_returns)
        if scale:
            RC = RC / RC.sum()

        return tools.to_array(RC)
