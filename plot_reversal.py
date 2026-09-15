"""Generate charts for the short-term reversal project."""
import pandas as pd
import matplotlib.pyplot as plt

# --- Chart 1: cumulative return, train/test split marked ---
cum = pd.read_csv("portfolio_cum_returns.csv", index_col=0, parse_dates=True).iloc[:, 0]
split_idx = int(len(cum) * 0.70)
split_date = cum.index[split_idx]

fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(cum.index, cum.values, color="#1f77b4", linewidth=1.2)
ax.axvline(split_date, color="grey", linestyle="--", linewidth=1)
ax.axhline(1.0, color="grey", linewidth=0.6, linestyle=":")
ax.text(split_date, ax.get_ylim()[1] * 0.98, " test period starts", fontsize=9, va="top")
ax.set_title("Short-term reversal portfolio: cumulative return")
ax.set_ylabel("Portfolio value (base 1.0)")
ax.grid(alpha=0.25)
fig.autofmt_xdate()
fig.tight_layout()
fig.savefig("cumulative_return.png", dpi=150)
print("saved cumulative_return.png")

# --- Chart 2: parameter robustness grid ---
robustness = pd.read_csv("parameter_robustness.csv")

fig2, ax2 = plt.subplots(figsize=(8, 5))
ax2.plot(robustness["formation_days"], robustness["train_sharpe"], marker="o",
          label="Train Sharpe", color="#2ca02c")
ax2.plot(robustness["formation_days"], robustness["test_sharpe"], marker="o",
          label="Test Sharpe", color="#d62728")
ax2.axhline(0, color="grey", linewidth=0.8, linestyle="--")
ax2.set_xlabel("Formation window (trading days)")
ax2.set_ylabel("Annualized Sharpe ratio")
ax2.set_title("Parameter robustness: does any formation window work?")
ax2.legend()
ax2.grid(alpha=0.25)
fig2.tight_layout()
fig2.savefig("parameter_robustness.png", dpi=150)
print("saved parameter_robustness.png")

# --- Chart 3: quintile breakdown ---
quintiles = pd.read_csv("quintile_breakdown.csv", index_col=0).iloc[:, 0]

fig3, ax3 = plt.subplots(figsize=(7, 5))
labels = ["Q1\n(biggest losers)", "Q2", "Q3", "Q4", "Q5\n(biggest winners)"]
colors = ["#2ca02c", "#7f7f7f", "#7f7f7f", "#7f7f7f", "#d62728"]
ax3.bar(labels, quintiles.values, color=colors)
ax3.set_ylabel("Annualized forward return")
ax3.set_title("Forward return by formation-week quintile\n(not monotonic = no clean reversal pattern)")
ax3.grid(axis="y", alpha=0.25)
fig3.tight_layout()
fig3.savefig("quintile_breakdown.png", dpi=150)
print("saved quintile_breakdown.png")
