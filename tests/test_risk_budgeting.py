import numpy as np

from pyrb.allocation import (
    ConstrainedRiskBudgeting,
    EqualRiskContribution,
    RiskBudgeting,
)

CORRELATION_MATRIX = np.array(
    [
        [1, 0.1, 0.4, 0.5, 0.5],
        [0.1, 1, 0.7, 0.4, 0.4],
        [0.4, 0.7, 1, 0.8, 0.05],
        [0.5, 0.4, 0.8, 1, 0.1],
        [0.5, 0.4, 0.05, 0.1, 1],
    ]
)
vol = [0.15, 0.20, 0.25, 0.3, 0.1]
NUMBEROFASSET = len(vol)
COVARIANCE_MATRIX = CORRELATION_MATRIX * np.outer(vol, vol)
RISKBUDGET = [0.2, 0.2, 0.3, 0.1, 0.2]
BOUNDS = np.array([[0.2, 0.3], [0.2, 0.3], [0.05, 0.15], [0.05, 0.15], [0.25, 0.35]])


def test_erc():
    ERC = EqualRiskContribution(COVARIANCE_MATRIX)
    ERC.solve()
    np.testing.assert_almost_equal(np.sum(ERC.weights), 1)
    np.testing.assert_almost_equal(
        np.dot(np.dot(ERC.weights, COVARIANCE_MATRIX), ERC.weights) ** 0.5,
        ERC.get_risk_contributions(scale=False).sum(),
        decimal=10,
    )
    np.testing.assert_equal(
        abs(ERC.get_risk_contributions().mean() - 1.0 / NUMBEROFASSET) < 1e-5, True
    )


def test_rb():
    RB = RiskBudgeting(COVARIANCE_MATRIX, RISKBUDGET)
    RB.solve()
    np.testing.assert_almost_equal(np.sum(RB.weights), 1, decimal=5)
    np.testing.assert_almost_equal(
        np.dot(np.dot(RB.weights, COVARIANCE_MATRIX), RB.weights) ** 0.5,
        RB.get_risk_contributions(scale=False).sum(),
        decimal=10,
    )
    np.testing.assert_equal(
        abs(RB.get_risk_contributions() - RISKBUDGET).sum() < 1e-5, True
    )


def test_cerb():
    CRB = ConstrainedRiskBudgeting(
        COVARIANCE_MATRIX, budgets=None, expected_returns=None, bounds=BOUNDS
    )
    CRB.solve()
    np.testing.assert_almost_equal(np.sum(CRB.weights), 1)
    np.testing.assert_almost_equal(CRB.get_risk_contributions()[1], 0.2455, decimal=5)
    np.testing.assert_almost_equal(np.sum(CRB.weights[1]), 0.2)
