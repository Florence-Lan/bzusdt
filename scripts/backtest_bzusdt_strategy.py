#!/usr/bin/env python3
"""Backtest a BZUSDT Brent crude swing strategy using public Binance Futures data."""

from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import urlopen

import pandas as pd


BASE_URL = "https://fapi.binance.com/fapi/v1/klines"


@dataclass(frozen=True)
class Params:
    lookback: int = 24
    dip_pct: float = 0.045
    rally_pct: float = 0.055
    rsi_long_max: float = 45.0
    rsi_short_min: float = 60.0
    vol_mult: float = 1.0
    max_atr_pct: float = 0.055
    stop_atr: float = 1.6
    take_atr: float = 2.4
    max_hold_bars: int = 18
    allow_short: bool = True


def fetch_klines(symbol: str, interval: str, limit: int) -> pd.DataFrame:
    url = f"{BASE_URL}?symbol={symbol}&interval={interval}&limit={limit}"
    with urlopen(url, timeout=20) as response:
        raw = json.loads(response.read().decode("utf-8"))
    if not raw or isinstance(raw, dict):
        raise RuntimeError(f"Unexpected Binance response: {raw}")

    columns = [
        "open_time",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "close_time",
        "quote_volume",
        "trade_count",
        "taker_buy_volume",
        "taker_buy_quote_volume",
        "ignore",
    ]
    df = pd.DataFrame(raw, columns=columns)
    numeric = ["open", "high", "low", "close", "volume", "quote_volume", "taker_buy_volume"]
    df[numeric] = df[numeric].astype(float)
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)
    df["trade_count"] = df["trade_count"].astype(int)
    return df


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    prev_close = out["close"].shift(1)
    tr = pd.concat(
        [
            out["high"] - out["low"],
            (out["high"] - prev_close).abs(),
            (out["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    out["atr"] = tr.rolling(14, min_periods=14).mean()
    out["atr_pct"] = out["atr"] / out["close"]
    out["ema_fast"] = out["close"].ewm(span=20, adjust=False).mean()
    out["ema_slow"] = out["close"].ewm(span=80, adjust=False).mean()
    delta = out["close"].diff()
    gain = delta.clip(lower=0).rolling(14, min_periods=14).mean()
    loss = (-delta.clip(upper=0)).rolling(14, min_periods=14).mean()
    rs = gain / loss.replace(0, math.nan)
    out["rsi"] = 100 - (100 / (1 + rs))
    out["vol_ma"] = out["volume"].rolling(20, min_periods=20).mean()
    out["vol_ratio"] = out["volume"] / out["vol_ma"]
    out["rolling_high"] = out["high"].rolling(24, min_periods=24).max().shift(1)
    out["rolling_low"] = out["low"].rolling(24, min_periods=24).min().shift(1)
    candle_range = (out["high"] - out["low"]).replace(0, math.nan)
    out["close_location"] = (out["close"] - out["low"]) / candle_range
    out["taker_buy_ratio"] = out["taker_buy_volume"] / out["volume"].replace(0, math.nan)
    return out


def signal_at(row: pd.Series, prev: pd.Series, params: Params) -> int:
    if pd.isna(row["atr"]) or pd.isna(row["rsi"]) or pd.isna(row["vol_ratio"]):
        return 0
    if row["atr_pct"] > params.max_atr_pct:
        return 0

    drawdown = (row["close"] / row["rolling_high"]) - 1 if row["rolling_high"] else 0
    rally = (row["close"] / row["rolling_low"]) - 1 if row["rolling_low"] else 0
    volume_ok = row["vol_ratio"] >= params.vol_mult

    absorption_long = (
        drawdown <= -params.dip_pct
        and row["rsi"] <= params.rsi_long_max
        and volume_ok
        and row["close"] > prev["close"]
        and row["close_location"] >= 0.55
    )
    trend_reclaim_long = (
        row["ema_fast"] > row["ema_slow"]
        and row["close"] > row["ema_fast"]
        and prev["close"] <= prev["ema_fast"]
        and volume_ok
        and row["taker_buy_ratio"] >= 0.50
    )
    failed_rally_short = (
        params.allow_short
        and rally >= params.rally_pct
        and row["rsi"] >= params.rsi_short_min
        and volume_ok
        and row["close"] < prev["close"]
        and row["close_location"] <= 0.45
    )

    if absorption_long or trend_reclaim_long:
        return 1
    if failed_rally_short:
        return -1
    return 0


def backtest(df: pd.DataFrame, params: Params, fee_bps: float, slippage_bps: float) -> tuple[dict, list[dict]]:
    cost = (fee_bps + slippage_bps) / 10_000
    equity = 1.0
    curve = []
    trades: list[dict] = []
    position = 0
    entry_price = 0.0
    entry_time = None
    entry_i = 0
    stop_price = 0.0
    take_price = 0.0

    for i in range(1, len(df) - 1):
        row = df.iloc[i]

        if position:
            exit_reason = None
            exit_price = None
            if position == 1:
                if row["low"] <= stop_price:
                    exit_reason = "stop"
                    exit_price = stop_price * (1 - cost)
                elif row["high"] >= take_price:
                    exit_reason = "take"
                    exit_price = take_price * (1 - cost)
            else:
                if row["high"] >= stop_price:
                    exit_reason = "stop"
                    exit_price = stop_price * (1 + cost)
                elif row["low"] <= take_price:
                    exit_reason = "take"
                    exit_price = take_price * (1 + cost)
            if exit_reason is None and i - entry_i >= params.max_hold_bars:
                exit_reason = "time"
                exit_price = row["close"] * (1 - cost if position == 1 else 1 + cost)

            if exit_reason is not None:
                trade_return = position * ((exit_price / entry_price) - 1)
                equity *= 1 + trade_return
                trades.append(
                    {
                        "entry_time": entry_time,
                        "exit_time": row["close_time"],
                        "side": "long" if position == 1 else "short",
                        "entry": entry_price,
                        "exit": exit_price,
                        "return_pct": trade_return * 100,
                        "reason": exit_reason,
                    }
                )
                position = 0

        if not position:
            sig = signal_at(row, df.iloc[i - 1], params)
            if sig:
                next_open = df.iloc[i + 1]["open"]
                atr = row["atr"]
                position = sig
                entry_i = i + 1
                entry_time = df.iloc[i + 1]["open_time"]
                entry_price = next_open * (1 + cost if sig == 1 else 1 - cost)
                if sig == 1:
                    stop_price = entry_price - params.stop_atr * atr
                    take_price = entry_price + params.take_atr * atr
                else:
                    stop_price = entry_price + params.stop_atr * atr
                    take_price = entry_price - params.take_atr * atr

        mark = row["close"]
        unrealized = 0.0 if not position else position * ((mark / entry_price) - 1)
        curve.append(equity * (1 + unrealized))

    if position:
        row = df.iloc[-1]
        exit_price = row["close"] * (1 - cost if position == 1 else 1 + cost)
        trade_return = position * ((exit_price / entry_price) - 1)
        equity *= 1 + trade_return
        trades.append(
            {
                "entry_time": entry_time,
                "exit_time": row["close_time"],
                "side": "long" if position == 1 else "short",
                "entry": entry_price,
                "exit": exit_price,
                "return_pct": trade_return * 100,
                "reason": "end",
            }
        )
        curve.append(equity)

    metrics = summarize(curve, trades)
    return metrics, trades


def summarize(curve: list[float], trades: list[dict]) -> dict:
    if not curve:
        curve = [1.0]
    series = pd.Series(curve)
    peak = series.cummax()
    max_dd = ((series / peak) - 1).min() * 100
    returns = [t["return_pct"] for t in trades]
    wins = [r for r in returns if r > 0]
    losses = [r for r in returns if r <= 0]
    profit_factor = sum(wins) / abs(sum(losses)) if losses else float("inf")
    return {
        "return_pct": (series.iloc[-1] - 1) * 100,
        "max_drawdown_pct": max_dd,
        "trades": len(trades),
        "win_rate_pct": (len(wins) / len(trades) * 100) if trades else 0.0,
        "avg_trade_pct": sum(returns) / len(returns) if returns else 0.0,
        "profit_factor": profit_factor,
    }


def grid_params() -> list[Params]:
    values = {
        "dip_pct": [0.035, 0.050],
        "rsi_long_max": [40.0, 45.0, 50.0],
        "vol_mult": [0.8, 1.1],
        "stop_atr": [1.2, 1.8],
        "take_atr": [2.0, 3.0],
        "max_hold_bars": [12, 24],
        "allow_short": [False, True],
    }
    keys = list(values)
    return [Params(**dict(zip(keys, combo))) for combo in itertools.product(*(values[k] for k in keys))]


def optimize(df: pd.DataFrame, fee_bps: float, slippage_bps: float) -> pd.DataFrame:
    split = max(int(len(df) * 0.70), 120)
    train = df.iloc[:split].reset_index(drop=True)
    test = df.iloc[split - 80 :].reset_index(drop=True)
    rows = []
    for params in grid_params():
        train_metrics, _ = backtest(train, params, fee_bps, slippage_bps)
        test_metrics, _ = backtest(test, params, fee_bps, slippage_bps)
        if train_metrics["trades"] < 3 or test_metrics["trades"] < 1:
            continue
        test_pf = test_metrics["profit_factor"]
        capped_test_pf = 3.0 if math.isinf(test_pf) else min(test_pf, 3.0)
        score = (
            test_metrics["return_pct"]
            + 0.25 * train_metrics["return_pct"]
            + 0.35 * capped_test_pf
            + 0.20 * test_metrics["win_rate_pct"]
            - 0.80 * abs(test_metrics["max_drawdown_pct"])
        )
        rows.append(
            {
                "score": score,
                "train_return_pct": train_metrics["return_pct"],
                "test_return_pct": test_metrics["return_pct"],
                "test_max_dd_pct": test_metrics["max_drawdown_pct"],
                "test_trades": test_metrics["trades"],
                "test_win_rate_pct": test_metrics["win_rate_pct"],
                "test_pf": test_metrics["profit_factor"],
                "params": params,
            }
        )
    return pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)


def fmt_metrics(title: str, metrics: dict) -> str:
    pf = metrics["profit_factor"]
    pf_text = "inf" if math.isinf(pf) else f"{pf:.2f}"
    return (
        f"{title}: return={metrics['return_pct']:.2f}% | "
        f"max_dd={metrics['max_drawdown_pct']:.2f}% | "
        f"trades={metrics['trades']} | win_rate={metrics['win_rate_pct']:.1f}% | "
        f"avg_trade={metrics['avg_trade_pct']:.2f}% | pf={pf_text}"
    )


def write_report(path: Path, df: pd.DataFrame, baseline: dict, best: dict, trades: list[dict], params: Params) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# BZUSDT Backtest Report",
        "",
        f"- Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}",
        f"- Data: {df.iloc[0]['open_time']} to {df.iloc[-1]['close_time']}",
        f"- Bars: {len(df)}",
        "",
        "## Baseline",
        "",
        f"- {fmt_metrics('Baseline', baseline)}",
        "",
        "## Improved Parameters",
        "",
        f"- `{params}`",
        f"- {fmt_metrics('Improved', best)}",
        "",
        "## Trades",
        "",
        "| Entry UTC | Exit UTC | Side | Entry | Exit | Return | Reason |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for trade in trades[-20:]:
        lines.append(
            f"| {trade['entry_time']} | {trade['exit_time']} | {trade['side']} | "
            f"{trade['entry']:.2f} | {trade['exit']:.2f} | {trade['return_pct']:.2f}% | {trade['reason']} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="BZUSDT")
    parser.add_argument("--interval", default="1h")
    parser.add_argument("--limit", type=int, default=1500)
    parser.add_argument("--fee-bps", type=float, default=5.0)
    parser.add_argument("--slippage-bps", type=float, default=2.0)
    parser.add_argument("--save-report", default="reports/backtest_summary.md")
    args = parser.parse_args()

    raw_df = fetch_klines(args.symbol, args.interval, args.limit)
    now = pd.Timestamp.now(tz="UTC")
    raw_df = raw_df[raw_df["close_time"] <= now].copy()
    df = add_indicators(raw_df).dropna().reset_index(drop=True)
    if len(df) < 200:
        print(f"Not enough bars after indicators: {len(df)}", file=sys.stderr)
        return 2

    baseline_params = Params()
    baseline_metrics, baseline_trades = backtest(df, baseline_params, args.fee_bps, args.slippage_bps)
    ranking = optimize(df, args.fee_bps, args.slippage_bps)
    if ranking.empty:
        print("No candidate produced enough trades.", file=sys.stderr)
        return 3

    best_params: Params = ranking.iloc[0]["params"]
    best_metrics, best_trades = backtest(df, best_params, args.fee_bps, args.slippage_bps)

    print(f"Data: {df.iloc[0]['open_time']} -> {df.iloc[-1]['close_time']} | bars={len(df)}")
    print(fmt_metrics("Baseline", baseline_metrics))
    print(fmt_metrics("Improved", best_metrics))
    print(f"Best params: {best_params}")
    print("\nTop candidates:")
    cols = ["score", "train_return_pct", "test_return_pct", "test_max_dd_pct", "test_trades", "test_win_rate_pct", "test_pf"]
    print(ranking[cols].head(8).to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    print("\nRecent trades:")
    for trade in best_trades[-10:]:
        print(
            f"{trade['entry_time']} -> {trade['exit_time']} {trade['side']:5s} "
            f"{trade['entry']:.2f}->{trade['exit']:.2f} {trade['return_pct']:.2f}% {trade['reason']}"
        )

    write_report(Path(args.save_report), df, baseline_metrics, best_metrics, best_trades, best_params)
    print(f"\nReport saved: {args.save_report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
