"""
Backtest statistics and metrics from a list of TradeRecord.
"""

from __future__ import annotations

import math
from typing import Any

from .mafilter_engine import TradeRecord


def compute_equity_curve(trades: list[TradeRecord], n_bars: int) -> list[float]:
    """Cumulative profit at end of each bar. Length n_bars+1 (equity at bar 0..n_bars)."""
    equity = [0.0] * (n_bars + 1)
    for b in range(1, n_bars + 1):
        equity[b] = sum(t.profit for t in trades if t.exit_bar < b)
    return equity


def compute_backtest_metrics(
    trades: list[TradeRecord],
    n_bars: int,
) -> dict[str, Any]:
    """
    Compute all final backtest panel metrics.
    Returns dict with: total_loss, total_profit, net_profit, n_long, n_short,
    n_long_won, n_short_won, n_long_lost, n_short_lost, avg_win_trade, avg_loss_trade,
    largest_win_trade, largest_loss_trade, profit_factor, sharpe_ratio, z_score,
    max_drawdown_abs, max_drawdown_pct.
    """
    if not trades:
        return {
            "total_loss": 0.0,
            "total_profit": 0.0,
            "net_profit": 0.0,
            "n_long": 0,
            "n_short": 0,
            "n_long_won": 0,
            "n_short_won": 0,
            "n_long_lost": 0,
            "n_short_lost": 0,
            "avg_win_trade": 0.0,
            "avg_loss_trade": 0.0,
            "avg_win_long_trade": 0.0,
            "avg_loss_long_trade": 0.0,
            "avg_win_short_trade": 0.0,
            "avg_loss_short_trade": 0.0,
            "largest_win_trade": 0.0,
            "largest_loss_trade": 0.0,
            "profit_factor": 0.0,
            "sharpe_ratio": 0.0,
            "z_score": 0.0,
            "max_drawdown_abs": 0.0,
            "max_drawdown_pct": 0.0,
            "total_trades": 0,
        }

    wins = [t.profit for t in trades if t.profit > 0]
    losses = [t.profit for t in trades if t.profit <= 0]
    total_profit = sum(wins)
    total_loss_abs = abs(sum(losses))
    net_profit = total_profit - total_loss_abs

    long_trades = [t for t in trades if t.direction == 1]
    short_trades = [t for t in trades if t.direction == -1]
    n_long = len(long_trades)
    n_short = len(short_trades)
    n_long_won = sum(1 for t in long_trades if t.profit > 0)
    n_short_won = sum(1 for t in short_trades if t.profit > 0)
    n_long_lost = n_long - n_long_won
    n_short_lost = n_short - n_short_won

    avg_win = (sum(wins) / len(wins)) if wins else 0.0
    avg_loss = (sum(losses) / len(losses)) if losses else 0.0
    largest_win = max(wins) if wins else 0.0
    largest_loss = min(losses) if losses else 0.0

    long_wins = [t.profit for t in long_trades if t.profit > 0]
    long_losses = [t.profit for t in long_trades if t.profit <= 0]
    short_wins = [t.profit for t in short_trades if t.profit > 0]
    short_losses = [t.profit for t in short_trades if t.profit <= 0]
    avg_win_long = (sum(long_wins) / len(long_wins)) if long_wins else 0.0
    avg_loss_long = (sum(long_losses) / len(long_losses)) if long_losses else 0.0
    avg_win_short = (sum(short_wins) / len(short_wins)) if short_wins else 0.0
    avg_loss_short = (sum(short_losses) / len(short_losses)) if short_losses else 0.0
    n_trades_total = n_long + n_short

    profit_factor = total_profit / total_loss_abs if total_loss_abs > 0 else (float("inf") if total_profit > 0 else 0.0)

    # Sharpe: assume each trade is one "period"; annualize with sqrt(252) if we had daily returns
    returns = [t.profit for t in trades]
    n_trades = len(returns)
    if n_trades < 2:
        sharpe_ratio = 0.0
    else:
        mean_ret = sum(returns) / n_trades
        variance = sum((r - mean_ret) ** 2 for r in returns) / (n_trades - 1)
        std_ret = math.sqrt(variance) if variance > 0 else 1e-12
        sharpe_ratio = (mean_ret / std_ret) * math.sqrt(252) if std_ret != 0 else 0.0

    # Z-Score (of trade sequence: wins = 1, losses = 0)
    win_loss_series = [1 if t.profit > 0 else 0 for t in trades]
    n_w = sum(win_loss_series)
    n_l = len(win_loss_series) - n_w
    if n_trades < 2:
        z_score = 0.0
    else:
        # Z = (W - 0.5*N) / sqrt(0.25*N)
        w = n_w
        n = n_trades
        z_score = (w - 0.5 * n) / math.sqrt(0.25 * n) if n > 0 else 0.0

    # Max DD: (Lowest Equity - Peak Equity) / Peak Equity × 100% (stored as negative)
    equity_curve = compute_equity_curve(trades, n_bars)
    peak = equity_curve[0]
    max_dd_abs = 0.0
    max_dd_pct = 0.0
    for eq in equity_curve:
        if eq > peak:
            peak = eq
        dd_abs = peak - eq
        if dd_abs > max_dd_abs:
            max_dd_abs = dd_abs
        dd_pct = (dd_abs / peak * 100.0) if peak > 0 else 0.0
        if dd_pct > max_dd_pct:
            max_dd_pct = dd_pct
    max_dd_pct = -max_dd_pct  # store as negative (drawdown = loss)

    return {
        "total_loss": total_loss_abs,
        "total_profit": total_profit,
        "net_profit": net_profit,
        "n_long": n_long,
        "n_short": n_short,
        "n_long_won": n_long_won,
        "n_short_won": n_short_won,
        "n_long_lost": n_long_lost,
        "n_short_lost": n_short_lost,
        "avg_win_trade": avg_win,
        "avg_loss_trade": avg_loss,
        "avg_win_long_trade": avg_win_long,
        "avg_loss_long_trade": avg_loss_long,
        "avg_win_short_trade": avg_win_short,
        "avg_loss_short_trade": avg_loss_short,
        "largest_win_trade": largest_win,
        "largest_loss_trade": largest_loss,
        "profit_factor": profit_factor,
        "sharpe_ratio": sharpe_ratio,
        "z_score": z_score,
        "max_drawdown_abs": max_dd_abs,
        "max_drawdown_pct": max_dd_pct,
        "total_trades": n_trades_total,
    }
