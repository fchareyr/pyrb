# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.2
#   kernelspec:
#     display_name: pyrb
#     language: python
#     name: python3
# ---

# %% [markdown]
# ## Example

# %%
import numpy as np
import pandas as pd

from pyrb import EqualRiskContribution, RiskBudgeting

# %%
# get a covariance matrix of an asset universe
covariance_matrix = (
    pd.read_csv("data.csv", sep=";", index_col=0).pct_change().cov() * 260
)

covariance_matrix

# %% [markdown]
# #### Solving the ERC problem

# %%
ERC = EqualRiskContribution(covariance_matrix)
ERC.solve()

# %% [markdown]
#  The optimal solution that gives equal risk contributions is:

# %%
optimal_weights = ERC.x
risk_contributions = ERC.get_risk_contributions(scale=False)
risk_contributions_scaled = ERC.get_risk_contributions()

COLUMNS = ["optimal weights", "risk contribution", "risk contribution (scaled)"]

allocation = pd.DataFrame(
    np.concatenate(
        [[optimal_weights, risk_contributions, risk_contributions_scaled]]
    ).T,
    index=covariance_matrix.index,
    columns=COLUMNS,
)

allocation_pct = allocation.copy()
allocation_pct["optimal weights"] = (allocation_pct["optimal weights"] * 100).round(2)
allocation_pct["risk contribution"] = (allocation_pct["risk contribution"] * 100).round(2)
allocation_pct["risk contribution (scaled)"] = (allocation_pct["risk contribution (scaled)"] * 100).round(2)
allocation_pct

# %% [markdown]
# #### Solving the risk budgeting problem
#

# %% [markdown]
# Now we want the risk contributions equal to specific budgets

# %%
budgets = [0.1, 0.1, 0.1, 0.2, 0.2, 0.05, 0.05, 0.05, 0.05, 0.1]
RB = RiskBudgeting(covariance_matrix, budgets)
RB.solve()

# %%
optimal_weights = RB.x
risk_contributions = RB.get_risk_contributions(scale=False)
risk_contributions_scaled = RB.get_risk_contributions()

allocation = pd.DataFrame(
    np.concatenate(
        [[optimal_weights, risk_contributions, risk_contributions_scaled]]
    ).T,
    index=covariance_matrix.index,
    columns=COLUMNS,
)

allocation_pct = allocation.copy()
allocation_pct["optimal weights"] = (allocation_pct["optimal weights"] * 100).round(2)
allocation_pct["risk contribution"] = (allocation_pct["risk contribution"] * 100).round(2)
allocation_pct["risk contribution (scaled)"] = (allocation_pct["risk contribution (scaled)"] * 100).round(2)
allocation_pct

# %%
