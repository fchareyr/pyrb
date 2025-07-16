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
# ### Repoducing Table 9 from the paper Constrained Risk Budgeting Portfolios.

# %%
import numpy as np

from pyrb import ConstrainedRiskBudgeting

# %%
vol = [0.05, 0.05, 0.07, 0.1, 0.15, 0.15, 0.15, 0.18]
corr = (
    np.array(
        [
            [100, 80, 60, -20, -10, -20, -20, -20],
            [80, 100, 40, -20, -20, -10, -20, -20],
            [60, 40, 100, 50, 30, 20, 20, 30],
            [-20, -20, 50, 100, 60, 60, 50, 60],
            [-10, -20, 30, 60, 100, 90, 70, 70],
            [-20, -10, 20, 60, 90, 100, 60, 70],
            [-20, -20, 20, 50, 70, 60, 100, 70],
            [-20, -20, 30, 60, 70, 70, 70, 100],
        ]
    )
    / 100
)

cov = np.outer(vol, vol) * corr

# %%
C = None
d = None

CRB = ConstrainedRiskBudgeting(cov, C=C, d=d)
CRB.solve()
print(CRB)

# %%
C = np.array([[0, 0, 0, 0, -1.0, -1.0, -1.0, -1.0]])
d = [-0.3]

CRB = ConstrainedRiskBudgeting(cov, C=C, d=d)
CRB.solve()
print(CRB)

# %%
C = np.array([[0, 0, 0, 0, -1.0, -1.0, -1.0, -1.0], [1, -1, 0, 0, 1, -1, 0, 0]])
d = [-0.3, -0.05]

CRB = ConstrainedRiskBudgeting(cov, C=C, d=d)
CRB.solve()
print(CRB)

# %%
C = np.array([[0, 0, 0, 0, -1.0, -1.0, -1.0, -1.0], [1, -1, 0, 0, 1, -1, 0, 0]])
d = [-0.3, -0.05]
pi = [-0.1, 0.05, 0, 0, 0.2, 0.1, 0, -0.1]  # not in the paper
c = 2

CRB = ConstrainedRiskBudgeting(cov, C=C, d=d, pi=pi, c=c)
CRB.solve()
print(CRB)

# %%
