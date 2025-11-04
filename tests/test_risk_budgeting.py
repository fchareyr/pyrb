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

NUMBER_OF_ASSET = len(vol)
COVARIANCE_MATRIX = CORRELATION_MATRIX * np.outer(vol, vol)
RISK_BUDGETS = [0.2, 0.2, 0.3, 0.1, 0.2]
EXPECTED_RETURNS = [-0.1, 0.05, 0, 0, 0.2]
RISK_AVERSION = 2
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
        abs(ERC.get_risk_contributions().mean() - 1.0 / NUMBER_OF_ASSET) < 1e-5, True
    )


def test_rb():
    RB = RiskBudgeting(COVARIANCE_MATRIX, RISK_BUDGETS)
    RB.solve()
    np.testing.assert_almost_equal(np.sum(RB.weights), 1, decimal=5)
    np.testing.assert_almost_equal(
        np.dot(np.dot(RB.weights, COVARIANCE_MATRIX), RB.weights) ** 0.5,
        RB.get_risk_contributions(scale=False).sum(),
        decimal=10,
    )
    np.testing.assert_equal(
        abs(RB.get_risk_contributions() - RISK_BUDGETS).sum() < 1e-5, True
    )


def test_cerb():
    CRB = ConstrainedRiskBudgeting(
        COVARIANCE_MATRIX, budgets=None, expected_returns=None, bounds=BOUNDS
    )
    CRB.solve()
    np.testing.assert_almost_equal(np.sum(CRB.weights), 1)
    np.testing.assert_almost_equal(CRB.get_risk_contributions()[1], 0.2455, decimal=5)
    np.testing.assert_almost_equal(np.sum(CRB.weights[1]), 0.2)


def test_rb_with_equal_budgets():
    equal_budgets = [1.0 / NUMBER_OF_ASSET] * NUMBER_OF_ASSET
    RB = RiskBudgeting(COVARIANCE_MATRIX, equal_budgets)
    RB.solve()
    np.testing.assert_almost_equal(np.sum(RB.weights), 1, decimal=5)
    np.testing.assert_almost_equal(
        np.dot(np.dot(RB.weights, COVARIANCE_MATRIX), RB.weights) ** 0.5,
        RB.get_risk_contributions(scale=False).sum(),
        decimal=10,
    )
    np.testing.assert_equal(
        abs(RB.get_risk_contributions() - equal_budgets).sum() < 1e-5, True
    )


def test_cerb_with_expected_returns():
    C = np.array([[0, 0, 0, -1.0, -1.0]])
    d = [-0.3]

    CRB = ConstrainedRiskBudgeting(
        COVARIANCE_MATRIX,
        budgets=RISK_BUDGETS,
        expected_returns=EXPECTED_RETURNS,
        bounds=BOUNDS,
        C=C,
        d=d,
    )
    CRB.solve()

    np.testing.assert_almost_equal(np.sum(CRB.weights), 1)
    np.testing.assert_almost_equal(
        np.dot(EXPECTED_RETURNS, CRB.weights), 0.061, decimal=3
    )
    np.testing.assert_almost_equal(CRB.get_risk_contributions()[2], 0.51, decimal=5)


def test_erc_with_scaled_risk_contributions():
    ERC = EqualRiskContribution(COVARIANCE_MATRIX)
    ERC.solve()
    scaled_risk_contributions = ERC.get_risk_contributions(scale=True)
    np.testing.assert_almost_equal(np.sum(scaled_risk_contributions), 1, decimal=5)
    np.testing.assert_equal(
        abs(scaled_risk_contributions.mean() - 1.0 / NUMBER_OF_ASSET) < 1e-5, True
    )
