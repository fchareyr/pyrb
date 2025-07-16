import logging
from abc import ABC, abstractmethod
from typing import Literal, Any

import numpy as np
import scipy.optimize as optimize

from . import tools, validation
from .settings import BISECTION_UPPER_BOUND, MAX_ITER_BISECTION
from .solvers import solve_rb_admm_ccd, solve_rb_admm_qp, solve_rb_ccd


class RiskBudgetAllocation(ABC):
    def __init__(
        self,
        cov: np.ndarray[Any, Any],
        expected_returns: np.ndarray[Any, Any] | None = None,
        x: np.ndarray[Any, Any] | None = None,
    ) -> None:
        """Base class for Risk Budgeting Allocation.

        Args:
            cov: Covariance matrix of the returns, shape (n, n).
            expected_returns: Expected excess return for each asset, shape (n,).
                The default is None which implies 0 for each asset.
            x: Array of weights, shape (n,).
        """
        self.__n = cov.shape[0]
        if x is None:
            x = np.array([np.nan] * self.n)
        self._x = x
        validation.check_covariance(cov)

        if expected_returns is None:
            expected_returns = np.array([0.0] * self.n)
        validation.check_expected_return(expected_returns, self.n)
        self.__expected_returns = tools.to_column_matrix(expected_returns)

        self.__cov = np.array(cov)
        self.llambda_star: float = np.nan

    @property
    def cov(self) -> np.ndarray[Any, Any]:
        return self.__cov

    @property
    def x(self) -> np.ndarray[Any, Any]:
        return self._x

    @property
    def expected_returns(self) -> np.ndarray[Any, Any]:
        return self.__expected_returns

    @property
    def n(self) -> int:
        return self.__n

    @abstractmethod
    def solve(self) -> None:
        """Solve the problem.

        This is an abstract method that must be implemented by subclasses.
        """
        pass

    @abstractmethod
    def get_risk_contributions(self, scale: bool = True) -> np.ndarray:
        """Get the risk contribution of the Risk Budgeting Allocation."""
        pass

    def get_variance(self) -> float:
        """Get the portfolio variance: x.T * cov * x.

        Returns:
            Portfolio variance as a float.
        """
        x = self.x
        cov = self.cov
        x = tools.to_column_matrix(x)
        cov = np.asarray(cov)
        rc = np.multiply(x, cov @ x)
        return float(np.sum(tools.to_array(rc)))

    def get_volatility(self) -> float:
        """Get the portfolio volatility: sqrt(x.T * cov * x).

        Returns:
            Portfolio volatility as a float.
        """
        return self.get_variance() ** 0.5

    def get_expected_return(self) -> float:
        """Get the portfolio expected excess returns: x.T * expected_returns.

        Returns:
            Portfolio expected excess return as a float, or NaN if expected_returns is None.
        """
        if self.expected_returns is None:
            return np.nan
        else:
            x = self.x
            x = tools.to_column_matrix(x)
        return float(x.T @ self.expected_returns)

    def __str__(self) -> str:
        return (
            f"solution x: {np.round(self.x * 100, 4)}\n"
            f"llambda star: {np.round(self.llambda_star * 100, 4)}\n"
            f"risk contributions: {np.round(self.get_risk_contributions() * 100, 4)}\n"
            f"sigma(x): {np.round(self.get_volatility() * 100, 4)}\n"
            f"sum(x): {np.round(self.x.sum() * 100, 4)}\n"
        )


class EqualRiskContribution(RiskBudgetAllocation):
    def __init__(self, cov: np.ndarray[Any, Any]) -> None:
        """Solve the equal risk contribution problem using cyclical coordinate descent.

        Although this does not change the optimal solution, the risk measure
        considered is the portfolio volatility.

        Args:
            cov: Covariance matrix of the returns, shape (n, n).
        """
        RiskBudgetAllocation.__init__(self, cov)

    def solve(self) -> None:
        """Solve the equal risk contribution problem using cyclical coordinate descent.

        Updates the internal weights (x) and llambda_star attributes.
        """
        x = solve_rb_ccd(cov=self.cov)
        if x is not None:
            self._x = tools.to_array(x / x.sum())
            self.llambda_star = self.get_volatility()

    def get_risk_contributions(self, scale: bool = True) -> np.ndarray:
        x = self.x
        cov = self.cov
        x = tools.to_column_matrix(x)
        cov = np.asarray(cov)
        rc = np.multiply(x, cov @ x) / self.get_volatility()
        if scale:
            rc = rc / rc.sum()
        return tools.to_array(rc)


class RiskBudgeting(RiskBudgetAllocation):
    def __init__(self, cov: np.ndarray[Any, Any], budgets: np.ndarray[Any, Any]) -> None:
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

    def solve(self) -> None:
        """Solve the risk budgeting problem using cyclical coordinate descent.

        Updates the internal weights (x) and llambda_star attributes.
        """
        x = solve_rb_ccd(cov=self.cov, budgets=self.budgets)
        if x is not None:
            self._x = tools.to_array(x / x.sum())
            self.llambda_star = self.get_volatility()

    def get_risk_contributions(self, scale: bool = True) -> np.ndarray:
        x = self.x
        cov = self.cov
        x = tools.to_column_matrix(x)
        cov = np.asarray(cov)
        rc = np.multiply(x, cov @ x) / self.get_volatility()
        if scale:
            rc = rc / rc.sum()
        return tools.to_array(rc)


class RiskBudgetingWithER(RiskBudgetAllocation):
    def __init__(
        self,
        cov: np.ndarray[Any, Any],
        budgets: np.ndarray[Any, Any] | None = None,
        expected_returns: np.ndarray[Any, Any] | None = None,
        risk_aversion: float = 1,
    ) -> None:
        """Solve the risk budgeting problem for the standard deviation risk measure.

        Uses cyclical coordinate descent. The risk measure is given by
        R(x) = c * sqrt(x^T cov x) - expected_returns^T x.

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

    def solve(self) -> None:
        x = solve_rb_ccd(
            cov=self.cov,
            budgets=self.budgets,
            expected_returns=self.expected_returns,
            risk_aversion=self.risk_aversion,
        )
        self._x = tools.to_array(x / x.sum())
        self.llambda_star = (
            -self.get_expected_return() + self.get_volatility() * self.risk_aversion
        )

    def get_risk_contributions(self, scale: bool = True) -> np.ndarray:
        x = self.x
        cov = self.cov
        x = tools.to_column_matrix(x)
        cov = np.asarray(cov)
        rc = (
            np.multiply(x, cov @ x) / self.get_volatility() * self.risk_aversion
            - self.x * self.expected_returns
        )
        if scale:
            rc = rc / rc.sum()
        return tools.to_array(rc)

    def __str__(self) -> str:
        return (
            super().__str__()
            + f"mu(x): {np.round(self.get_expected_return() * 100, 4)}\n"
        )


class ConstrainedRiskBudgeting(RiskBudgetingWithER):
    def __init__(
        self,
        cov: np.ndarray[Any, Any],
        budgets: np.ndarray[Any, Any] | None = None,
        expected_returns: np.ndarray[Any, Any] | None = None,
        risk_aversion: float = 1,
        inequality_constraints: np.ndarray[Any, Any] | None = None,
        inequality_values: np.ndarray[Any, Any] | None = None,
        bounds: np.ndarray[Any, Any] | None = None,
        solver: Literal["admm_ccd", "admm_qp"] = "admm_ccd",
    ) -> None:
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
            inequality_constraints: Array of p inequality constraints, shape (p, n). If None the
                problem is unconstrained and solved using CCD (algorithm 3)
                and it solves equation (17).
            inequality_values: Array of p constraints that matches the inequalities, shape (p,).
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

        self.inequality_values = inequality_values
        self.inequality_constraints = inequality_constraints
        self.bounds = bounds
        validation.check_bounds(bounds, self.n)
        validation.check_constraints(inequality_constraints, inequality_values, self.n)
        self.solver = solver
        if (self.solver == "admm_qp") and (self.expected_returns is not None):
            logging.warning(
                "The solver is set to 'admm_qp'. The risk measure is the mean variance in this case. The optimal "
                "solution will not be the same than 'admm_ccd' when expected_returns is not zero.     "
            )

    @property
    def C(self) -> np.ndarray[Any, Any] | None:  # noqa: N802
        return self.inequality_constraints

    @property
    def d(self) -> np.ndarray[Any, Any] | None:
        return self.inequality_values

    def __str__(self) -> str:
        if self.C is not None:
            return (
                f"solver: {self.solver}\n"
                + "----------------------------\n"
                + super().__str__()
                + f"C@x: {self.C @ self.x}\n"
            )
        else:
            return super().__str__()

    def _sum_to_one_constraint(self, llambda: float) -> float:
        x = self._llambda_solve(llambda)
        sum_x = sum(x)
        return sum_x - 1

    def _llambda_solve(self, llambda: float) -> np.ndarray:
        if (
            self.inequality_constraints is None
        ):  # it is optimal to take the CCD in case of separable constraints
            x = solve_rb_ccd(
                self.cov,
                self.budgets,
                self.expected_returns,
                self.risk_aversion,
                self.bounds,
                llambda,
            )
            self.solver = "ccd"
        elif self.solver == "admm_qp":
            x = solve_rb_admm_qp(
                cov=self.cov,
                budgets=self.budgets,
                expected_returns=self.expected_returns,
                risk_aversion=self.risk_aversion,
                C=self.C,
                d=self.d,
                bounds=self.bounds,
                llambda_log=llambda,
            )
        elif self.solver == "admm_ccd":
            x = solve_rb_admm_ccd(
                cov=self.cov,
                budgets=self.budgets,
                expected_returns=self.expected_returns,
                risk_aversion=self.risk_aversion,
                C=self.C,
                d=self.d,
                bounds=self.bounds,
                llambda_log=llambda,
            )
        else:
            raise ValueError(f"Unknown solver: {self.solver}")
        return x

    def solve(self) -> None:
        try:
            llambda_star = optimize.bisect(
                self._sum_to_one_constraint,
                0,
                BISECTION_UPPER_BOUND,
                maxiter=MAX_ITER_BISECTION,
            )
            self.llambda_star = llambda_star
            self._x = self._llambda_solve(llambda_star)
        except Exception as e:
            if e.args[0] == "f(a) and f(b) must have different signs":
                logging.exception(
                    "Bisection failed: "
                    + str(e)
                    + ". If you are using expected returns the parameter 'c' need to be correctly scaled (see remark 1 in the paper). Otherwise please check the constraints or increase the bisection upper bound."
                )
            else:
                logging.exception("Problem not solved: " + str(e))

    def get_risk_contributions(self, scale: bool = True) -> np.ndarray:
        """Return the risk contribution.

        If the solver is "admm_qp" the mean variance risk measure is considered.

        Args:
            scale: If True, the sum on risk contribution is scaled to one.

        Returns:
            Risk contribution of each asset, shape (n,).
        """
        x = self.x
        cov = self.cov
        x = tools.to_column_matrix(x)
        cov = np.asarray(cov)

        if self.solver == "admm_qp":
            rc = np.multiply(x, cov @ x) - self.risk_aversion * self.x * self.expected_returns
        else:
            rc = np.multiply(x, cov @ x).T / self.get_volatility() * self.risk_aversion - tools.to_array(
                self.x.T
            ) * tools.to_array(self.expected_returns)
        if scale:
            rc = rc / rc.sum()

        return tools.to_array(rc)
