"""
Short-term reversal: a cross-sectional mean-reversion strategy on 40
liquid US large-cap stocks, with a full P&L backtest and the statistical
checks needed to tell a real edge from noise.

Idea (Jegadeesh (1990), "short-term reversal" - one of the most
replicated cross-sectional equity factors): stocks that fell the most
over the past week tend to partially bounce back over the following
week, and stocks that rose the most tend to partially give some back -
consistent with short-term overreaction and liquidity provision by
market makers absorbing order flow. This is a genuinely different
mean-reversion mechanism from the pairs-trading project elsewhere in this
series (which bets on a *pair's spread* reverting, based on cointegration
between two specific names) - here the bet is cross-sectional: at every
rebalance, rank ALL 40 stocks against each other and trade the extremes.

Construction:
  1. Formation: each week, rank all 40 stocks by their trailing return
     over the past `FORMATION_DAYS` trading days.
  2. Portfolio: go long the bottom quintile (biggest recent losers) and
     short the top quintile (biggest recent winners), equal-weighted
     within each leg, dollar-neutral (long leg - short leg).
  3. Holding period: 1 week, then re-rank and rebalance.
  4. Position applied with a 1-day lag after formation (no look-ahead:
     the ranking uses only data available before the position is taken).

Costs: 5 bps per leg on turnover at each rebalance (a conservative but
not punitive assumption for large-cap US equities).

Rigor beyond a simple train/test split:
  - Statistical significance of the Sharpe ratio (Lo (2002) asymptotic
    SE) and a block bootstrap confidence interval.
  - A parameter-robustness grid over the formation window (1, 2, 3, 4
    weeks) - short-term reversal is known in the literature to be
    strongest at short (1-week) horizons and to fade or reverse at longer
    ones, so this grid is a real test of that claim, not a formality.
  - A quintile-by-quintile breakdown (report every quintile's return, not
    just the two extremes), to check the effect is monotonic across the
    ranking rather than an artifact of just the two tail buckets.
"""

import numpy as np
import pandas as pd
import yfinance as yf

UNIVERSE = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "JPM", "BAC", "WFC",
    "GS", "MS", "XOM", "CVX", "COP", "JNJ", "PFE", "UNH", "MRK", "ABBV",
    "KO", "PEP", "WMT", "PG", "HD", "LOW", "DIS", "NFLX", "CSCO", "INTC",
    "IBM", "ORCL", "CRM", "ADBE", "QCOM", "TXN", "CAT", "BA", "GE", "MMM",
]

FORMATION_DAYS_DEFAULT = 5  # 1 trading week
REBALANCE_DAYS = 5          # weekly rebalance
QUINTILE = 0.2               # top/bottom 20%
TXN_COST_BPS = 5 / 10000
TRAIN_FRAC = 0.70
ANNUALIZATION = 252  # the backtest produces a DAILY return series (positions
                      # are held for a week but returns are marked to market
                      # daily), so annualization must use trading days, not
                      # the weekly rebalance frequency
N_BOOTSTRAP = 2000
BLOCK_SIZE = 40  # trading days (~8 weeks), for the block bootstrap


def load_prices():
    px = yf.download(UNIVERSE, period="10y", progress=False)["Close"]
    return px.dropna()


def weekly_rebalance_dates(px):
    return px.index[::REBALANCE_DAYS]


def build_reversal_positions(px, formation_days=FORMATION_DAYS_DEFAULT):
    """At each rebalance date, rank stocks by trailing `formation_days`
    return (using data strictly before the rebalance date), and assign
    -1/N_short to the top quintile (winners, to short) and +1/N_long to
    the bottom quintile (losers, to go long).
    """
    rebal_dates = weekly_rebalance_dates(px)
    positions = pd.DataFrame(0.0, index=px.index, columns=px.columns)

    for i, date in enumerate(rebal_dates):
        loc = px.index.get_loc(date)
        if loc < formation_days:
            continue
        formation_ret = px.iloc[loc] / px.iloc[loc - formation_days] - 1
        formation_ret = formation_ret.dropna()
        n = len(formation_ret)
        n_leg = max(1, int(n * QUINTILE))

        ranked = formation_ret.sort_values()
        losers = ranked.index[:n_leg]     # biggest fallers -> go long
        winners = ranked.index[-n_leg:]   # biggest risers -> go short

        next_loc = loc + 1
        end_loc = rebal_dates[i + 1] if i + 1 < len(rebal_dates) else px.index[-1]
        end_iloc = px.index.get_loc(end_loc)

        positions.iloc[next_loc:end_iloc + 1, positions.columns.get_indexer(losers)] = 1.0 / n_leg
        positions.iloc[next_loc:end_iloc + 1, positions.columns.get_indexer(winners)] = -1.0 / n_leg

    return positions


def backtest(px, positions):
    daily_ret = px.pct_change()
    gross_ret = (positions.shift(1) * daily_ret).sum(axis=1)

    turnover = positions.diff().abs().sum(axis=1).fillna(0)
    cost = turnover * TXN_COST_BPS
    net_ret = (gross_ret - cost).dropna()
    return net_ret


def sharpe_and_se(returns, annualization=ANNUALIZATION):
    n = len(returns)
    sr_period = returns.mean() / returns.std()
    sr_annual = sr_period * np.sqrt(annualization)
    se_annual = np.sqrt((1 + 0.5 * sr_period**2) / n) * np.sqrt(annualization)
    t_stat = sr_annual / se_annual
    return sr_annual, se_annual, t_stat, n


def block_bootstrap_sharpe_ci(returns, n_boot=N_BOOTSTRAP, block_size=BLOCK_SIZE, seed=42):
    rng = np.random.default_rng(seed)
    values = returns.values
    n = len(values)
    n_blocks = int(np.ceil(n / block_size))
    boot_sharpes = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, max(1, n - block_size), size=n_blocks)
        sample = np.concatenate([values[i:i + block_size] for i in idx])[:n]
        sr = sample.mean() / sample.std() * np.sqrt(ANNUALIZATION)
        boot_sharpes[b] = sr
    lo, hi = np.percentile(boot_sharpes, [2.5, 97.5])
    return lo, hi


def summarize(returns):
    cum = (1 + returns).cumprod()
    running_max = cum.cummax()
    max_dd = (cum / running_max - 1).min()
    sr, se, t, n = sharpe_and_se(returns)
    return {
        "n_days": n, "annualized_sharpe": sr, "sharpe_se": se, "t_stat": t,
        "max_drawdown": max_dd, "total_return": cum.iloc[-1] - 1,
    }


def parameter_robustness(px, split_idx, formation_grid=(3, 5, 10, 15, 20)):
    rows = []
    for fd in formation_grid:
        positions = build_reversal_positions(px, formation_days=fd)
        ret = backtest(px, positions)
        train, test = ret.iloc[:split_idx], ret.iloc[split_idx:]
        sr_train, *_ = sharpe_and_se(train) if len(train) > 60 else (np.nan,) * 4
        sr_test, *_ = sharpe_and_se(test) if len(test) > 60 else (np.nan,) * 4
        rows.append({"formation_days": fd, "train_sharpe": sr_train, "test_sharpe": sr_test})
    return pd.DataFrame(rows)


def quintile_breakdown(px, formation_days=FORMATION_DAYS_DEFAULT, n_quintiles=5):
    """Return, for each of the 5 quintiles (1=biggest losers .. 5=biggest
    winners), that quintile's OWN forward weekly return - checking whether
    the reversal effect is monotonic across the ranking, not just an
    artifact of the two tail buckets used in the main strategy.
    """
    rebal_dates = weekly_rebalance_dates(px)
    quintile_rets = {q: [] for q in range(1, n_quintiles + 1)}

    for i, date in enumerate(rebal_dates[:-1]):
        loc = px.index.get_loc(date)
        if loc < formation_days:
            continue
        formation_ret = (px.iloc[loc] / px.iloc[loc - formation_days] - 1).dropna()
        ranked = formation_ret.sort_values()
        n = len(ranked)
        bounds = np.linspace(0, n, n_quintiles + 1).astype(int)

        next_date = rebal_dates[i + 1]
        fwd_ret = px.loc[next_date] / px.loc[date] - 1

        for q in range(1, n_quintiles + 1):
            names = ranked.index[bounds[q - 1]:bounds[q]]
            quintile_rets[q].append(fwd_ret[names].mean())

    # these are weekly forward returns (one per rebalance), so annualize
    # with 52, independent of the daily ANNUALIZATION used for the main
    # backtest's Sharpe calculations
    return pd.Series({q: np.nanmean(v) * 52 for q, v in quintile_rets.items()})


def main():
    px = load_prices()
    positions = build_reversal_positions(px)
    portfolio_ret = backtest(px, positions)

    n = len(portfolio_ret)
    split = int(n * TRAIN_FRAC)
    train, test = portfolio_ret.iloc[:split], portfolio_ret.iloc[split:]

    train_stats = summarize(train)
    test_stats = summarize(test)
    stats_df = pd.DataFrame([{"period": "train", **train_stats},
                              {"period": "test", **test_stats}]).set_index("period")
    print("=== Train/test performance ===")
    print(stats_df.to_string(float_format=lambda x: f"{x:.4f}"))
    stats_df.to_csv("train_test_stats.csv")

    lo, hi = block_bootstrap_sharpe_ci(test)
    print(f"\n95% block-bootstrap CI for test-period annualized Sharpe: [{lo:.3f}, {hi:.3f}]")
    pd.Series({"ci_low": lo, "ci_high": hi}).to_csv("bootstrap_ci.csv")

    robustness = parameter_robustness(px, split)
    print("\n=== Parameter robustness (formation window, trading days) ===")
    print(robustness.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    robustness.to_csv("parameter_robustness.csv", index=False)

    quintiles = quintile_breakdown(px)
    print("\n=== Quintile breakdown (1=biggest losers, 5=biggest winners) ===")
    print("Annualized forward return by quintile:")
    print(quintiles.to_string(float_format=lambda x: f"{x:.4f}"))
    quintiles.to_csv("quintile_breakdown.csv")

    portfolio_ret.to_csv("portfolio_returns.csv")
    (1 + portfolio_ret).cumprod().to_csv("portfolio_cum_returns.csv")

    return stats_df, robustness, quintiles


if __name__ == "__main__":
    main()
