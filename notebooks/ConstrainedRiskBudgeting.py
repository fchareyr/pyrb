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

# %% [markdown] vscode={"languageId": "latex"}
# # Introduction
#
# This notebook reproduces (a subset of) Table 9 from Richard & Roncalli (2019) on Constrained Risk Budgeting portfolios.
#
# We:
# 1. Define an 8-asset universe via individual volatilities and a correlation matrix (converted to a covariance matrix `cov`).
# 2. Solve a sequence of Constrained Risk Budgeting (CRB) problems using the `ConstrainedRiskBudgeting` class from `pyrb`.
# 3. Start with the unconstrained risk budgeting solution (no linear constraints).
# 4. Incrementally add linear inequality constraints of the form C w ≥ d:
#     - First constraint: caps the aggregate allocation to the last four (higher-vol) assets.
#     - Second constraint: adds a relative allocation condition between two asset groups.
# 5. Compare how the optimal weights (and implicitly risk contributions) adjust as constraints tighten.
#
# Goal: illustrate how linear constraints reshape a risk budgeting allocation while maintaining the risk-budgeting structure as much as possible.

# %% [markdown]
# ## Optimization framing
# We consider risk budgeting portfolios with weights $w \in \mathbb{R}^n$ over $n=8$ assets and covariance matrix $\Sigma = \texttt{cov}$. The unconstrained risk budgeting problem seeks weights whose marginal risk contributions are proportional to a target budget vector $b$ (equal budgets by default):
# $$\begin{aligned}
# \text{find } & w \\
# \text{s.t. } & \sum_{i=1}^n w_i = 1, \quad w_i \ge 0, \\
# & w_i \cdot (\Sigma w)_i = b_i \cdot \sigma(w) \quad \text{for } i = 1,\dots,n,
# \end{aligned}$$
# where $\sigma(w) = \sqrt{w^\top \Sigma w}$ is the total portfolio risk. The `ConstrainedRiskBudgeting` solver enforces these equalized contributions while accommodating additional linear constraints of the form
# $$C w \ge d,$$
# with $C \in \mathbb{R}^{p \times n}$ and $d \in \mathbb{R}^p$.

# %% [markdown]
# ### Scenario summary
# - **Scenario 1 – Unconstrained:** solve the base risk budgeting problem with only the simplex bounds $w_i \ge 0$ and $\sum_i w_i = 1$.
# - **Scenario 2 – Cap high-vol bucket:** augment the base problem with
#   $$\sum_{i=5}^8 w_i \le 0.30,$$
#   limiting total allocation to the four higher-volatility assets.
# - **Scenario 3 – Add relative allocation:** keep the high-vol cap and require
#   $$w_1 - w_2 + w_5 - w_6 \ge -0.05,$$
#   ensuring the combined exposure to assets 1 and 5 does not fall too far below that of assets 2 and 6.

# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pyrb import ConstrainedRiskBudgeting

plt.style.use("tableau-colorblind10")
plt.rcParams.update({"figure.autolayout": True, "axes.grid": True})

# %%
vol = np.array([0.05, 0.05, 0.07, 0.1, 0.15, 0.15, 0.15, 0.18])
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
asset_labels = [f"A{i + 1}" for i in range(len(vol))]

# %%
scenario_specs = [
    {
        "name": "Unconstrained",
        "description": "No linear constraints.",
        "C": None,
        "d": None,
    },
    {
        "name": "Cap high-vol bucket",
        "description": "Sum of assets 5–8 constrained to be ≤ 30%.",
        "C": np.array([[0.0, 0.0, 0.0, 0.0, -1.0, -1.0, -1.0, -1.0]]),
        "d": np.array([-0.3]),
    },
    {
        "name": "Add relative allocation",
        "description": "Adds w1 - w2 + w5 - w6 ≥ -5% on top of the high-vol cap.",
        "C": np.array(
            [
                [0.0, 0.0, 0.0, 0.0, -1.0, -1.0, -1.0, -1.0],
                [1.0, -1.0, 0.0, 0.0, 1.0, -1.0, 0.0, 0.0],
            ]
        ),
        "d": np.array([-0.3, -0.05]),
    },
]


# %%
def solve_crb_scenario(cov_matrix, spec):
    model = ConstrainedRiskBudgeting(cov=cov_matrix, C=spec["C"], d=spec["d"])
    model.solve()
    weights = np.array(model.weights, dtype=float)
    weights = weights / weights.sum()
    volatility = float(model.get_volatility())
    marginal = (cov_matrix @ weights) / volatility
    risk_contrib = weights * marginal
    risk_contrib_pct = risk_contrib / risk_contrib.sum()
    constraint_projection = (spec["C"] @ weights) if spec["C"] is not None else None
    return {
        "name": spec["name"],
        "description": spec["description"],
        "weights": weights,
        "marginal_rc": marginal,
        "risk_contrib": risk_contrib,
        "risk_contrib_pct": risk_contrib_pct,
        "volatility": volatility,
        "lambda_star": float(model.lambda_star),
        "constraint_values": constraint_projection,
        "sum_weights": float(weights.sum()),
    }


scenario_results = [solve_crb_scenario(cov, spec) for spec in scenario_specs]

# %%
summary_df = pd.DataFrame(
    [
        {
            "Scenario": res["name"],
            "Description": res["description"],
            "Total risk (volatility)": res["volatility"],
            "λ*": res["lambda_star"],
            "Sum of weights": res["sum_weights"],
        }
        for res in scenario_results
    ]
).set_index("Scenario")

weights_df = pd.DataFrame(
    [res["weights"] for res in scenario_results],
    columns=asset_labels,
    index=summary_df.index,
)

marginal_df = pd.DataFrame(
    [res["marginal_rc"] for res in scenario_results],
    columns=asset_labels,
    index=summary_df.index,
)

risk_contrib_df = pd.DataFrame(
    [res["risk_contrib"] for res in scenario_results],
    columns=asset_labels,
    index=summary_df.index,
)

risk_contrib_pct_df = pd.DataFrame(
    [res["risk_contrib_pct"] for res in scenario_results],
    columns=asset_labels,
    index=summary_df.index,
)

constraint_df = pd.DataFrame(
    [
        pd.Series(res["constraint_values"], dtype=float)
        if res["constraint_values"] is not None
        else pd.Series(dtype=float)
        for res in scenario_results
    ],
    index=summary_df.index,
)

summary_display = summary_df.copy()
for col in ["Total risk (volatility)", "λ*", "Sum of weights"]:
    summary_display[col] = summary_display[col].astype(float).round(4)


def format_df(df, formatter):
    return df.apply(lambda col: col.map(formatter))


weights_display = format_df(weights_df, lambda x: f"{x:.2%}")
risk_contrib_display = format_df(risk_contrib_df, lambda x: f"{x:.4f}")
risk_contrib_pct_display = format_df(risk_contrib_pct_df, lambda x: f"{x:.2%}")
marginal_display = format_df(marginal_df, lambda x: f"{x:.4f}")

print("Summary metrics")
display(summary_display)

print("\nWeights")
display(weights_display)

print("\nRisk contributions (absolute)")
display(risk_contrib_display)

print("\nRisk contributions (share of total risk)")
display(risk_contrib_pct_display)

print("\nMarginal risk contributions")
display(marginal_display)

if not constraint_df.empty:
    constraint_display = constraint_df.copy().astype(float).round(4)
    constraint_display.columns = [
        f"Constraint {i + 1}" for i in range(constraint_display.shape[1])
    ]
    print("\nConstraint evaluations Cw")
    display(constraint_display)

# %%
fig, axes = plt.subplots(3, 1, figsize=(12, 16))

weights_df.plot(kind="bar", ax=axes[0])
axes[0].set_title("Portfolio weights by scenario")
axes[0].set_ylabel("Weight")
axes[0].legend(title="Asset", bbox_to_anchor=(1.02, 1), loc="upper left")

risk_contrib_pct_df.plot(kind="bar", ax=axes[1])
axes[1].set_title("Relative risk contributions")
axes[1].set_ylabel("Share of total risk")
axes[1].legend(title="Asset", bbox_to_anchor=(1.02, 1), loc="upper left")

marginal_df.plot(kind="bar", ax=axes[2])
axes[2].set_title("Marginal risk contributions")
axes[2].set_ylabel("∂σ/∂w")
axes[2].legend(title="Asset", bbox_to_anchor=(1.02, 1), loc="upper left")

for ax in axes:
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=0)

plt.show()

fig2, ax2 = plt.subplots(figsize=(8, 5))
summary_df["Total risk (volatility)"].plot(kind="bar", ax=ax2, color="#4c72b0")
ax2.set_title("Total portfolio risk across scenarios")
ax2.set_ylabel("Volatility")
ax2.set_xlabel("")
for container in ax2.containers:
    ax2.bar_label(container, fmt="{:.4f}")

plt.show()
