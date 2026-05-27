"""Compute real, re-runnable numbers behind the poster/report claims.

Produces (all written to outputs/):
  trades_is.csv, trades_oos.csv          — canonical simulation
  bench_results.json                     — every derived number used downstream

Covers: SPY matched-window beta/alpha, SPY buy-and-hold vs rule equity,
portfolio-level random-entry Monte Carlo, expiry vs non-expiry premium.
All from capstone data + modules only (no personal_research inputs).
"""
import json
import os
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
from scipy import stats

from modules.config.capstone_v4_params import (
    RSI_THRESHOLD, MIN_CONSECUTIVE, HOLD_DAYS, COMMISSION,
    IS_START, IS_END, OOS_START, RANDOM_SEED,
)
from modules.data import DataLoader, ExpiryCalculator
from modules.features import TechnicalIndicators, FeaturePipeline
from modules.evaluation.portfolio_simulator import run_simulation
from modules.evaluation.metrics import calculate_composite_score, calculate_sharpe_ratio

ROOT = Path(__file__).resolve().parent.parent
CACHE = str(ROOT / "cache")
CONST = ROOT / "cache" / "constituent_data"
OUT = Path(__file__).resolve().parent
OOS_END = "2026-05-22"


def metrics(rets):
    rets = np.asarray(rets, float)
    n = len(rets)
    wr = float((rets > 0).mean())
    gp = rets[rets > 0].sum()
    gl = abs(rets[rets < 0].sum())
    pf = gp / gl if gl > 0 else float("inf")
    sharpe = float(calculate_sharpe_ratio(rets * 100))
    cum = (1 + rets).cumprod()
    mdd = float(abs(((cum - np.maximum.accumulate(cum)) / np.maximum.accumulate(cum)).min()))
    comp = float(calculate_composite_score(pf, sharpe, wr, mdd))
    return dict(n=n, win_rate=wr, pf=float(pf), sharpe=sharpe, mdd=mdd, composite=comp)


def cum_and_mdd(daily_ret):
    cum = (1 + np.asarray(daily_ret, float)).cumprod()
    mdd = float(abs(((cum - np.maximum.accumulate(cum)) / np.maximum.accumulate(cum)).min()))
    return float(cum[-1] - 1), mdd


def get_trades():
    f_is, f_oos = OUT / "trades_is.csv", OUT / "trades_oos.csv"
    if f_is.exists() and f_oos.exists():
        return (pd.read_csv(f_is, parse_dates=["signal_date", "entry_date", "exit_date"]),
                pd.read_csv(f_oos, parse_dates=["signal_date", "entry_date", "exit_date"]))
    print("running IS simulation ...")
    tis = run_simulation(RSI_THRESHOLD, MIN_CONSECUTIVE, HOLD_DAYS,
                         start_date=IS_START, end_date=IS_END, cache_dir=CACHE, verbose=False)
    print("running OOS simulation ...")
    toos = run_simulation(RSI_THRESHOLD, MIN_CONSECUTIVE, HOLD_DAYS,
                          start_date=OOS_START, end_date=OOS_END, cache_dir=CACHE, verbose=False)
    tis.to_csv(f_is, index=False)
    toos.to_csv(f_oos, index=False)
    return tis, toos


def load_spy():
    spy = pd.read_csv(CONST / "SPY.csv", parse_dates=["Date"]).sort_values("Date")
    spy = spy.set_index("Date")
    return spy


def spy_benchmark(trades, spy):
    """Matched-window per-trade alpha + buy-and-hold vs rule equity."""
    t = trades.sort_values("entry_date").copy()
    spy_idx = spy.index
    stock_g, spy_g, net = [], [], []
    for _, r in t.iterrows():
        ed, xd = r["entry_date"], r["exit_date"]
        # SPY entry at open on entry_date, exit at close on exit_date (matched window)
        ei = spy_idx.searchsorted(ed)
        xi = spy_idx.searchsorted(xd)
        if ei >= len(spy_idx) or xi >= len(spy_idx):
            continue
        spy_entry = spy.iloc[ei]["Open"]
        spy_exit = spy.iloc[xi]["Close"]
        spy_g.append(spy_exit / spy_entry - 1)
        stock_g.append(r["gross_return"])
        net.append(r["net_return"])
    stock_g, spy_g, net = np.array(stock_g), np.array(spy_g), np.array(net)
    alpha = stock_g - spy_g
    tt = stats.ttest_rel(stock_g, spy_g)
    # buy-and-hold SPY over the rule's active span (IS)
    span = spy[(spy.index >= t["entry_date"].min()) & (spy.index <= t["exit_date"].max())]
    spy_ret = span["Close"].pct_change().fillna(0).values
    spy_cum, spy_mdd = cum_and_mdd(spy_ret)
    n_years = (span.index[-1] - span.index[0]).days / 365.25
    spy_cagr = (1 + spy_cum) ** (1 / n_years) - 1
    rule_eq = metrics(t["net_return"].values)
    rule_cum = float((1 + t["net_return"].values).cumprod()[-1] - 1)
    # fair window: rule trades whose entry falls inside the SPY data span (2018+)
    tw = t[t["entry_date"] >= span.index[0]]
    rule_cum_fair, rule_mdd_fair = cum_and_mdd(tw["net_return"].values)
    # matched, capital-comparable cumulative (rule's picks net vs SPY on the same windows)
    full = spy["Close"]
    full_cum = float(full.iloc[-1] / full.iloc[0] - 1)
    full_yrs = (full.index[-1] - full.index[0]).days / 365.25
    return {
        "rule_picks_matched_cum": float((1 + net).cumprod()[-1] - 1),
        "spy_matched_cum": float((1 + spy_g).cumprod()[-1] - 1),
        "rule_beats_spy_window_pct": float((net > spy_g).mean()),
        "spy_full_cum": full_cum,
        "spy_full_cagr": float((1 + full_cum) ** (1 / full_yrs) - 1),
        "rule_cum_fair_2018": rule_cum_fair,
        "rule_mdd_fair_2018": rule_mdd_fair,
        "n_trades_fair": int(len(tw)),
        "n_matched": int(len(stock_g)),
        "stock_mean_gross": float(stock_g.mean()),
        "spy_mean_matched": float(spy_g.mean()),
        "alpha_mean": float(alpha.mean()),
        "alpha_t": float(tt.statistic),
        "alpha_p": float(tt.pvalue),
        "alpha_after_cost_mean": float(alpha.mean() - COMMISSION),
        "spy_span_start": str(span.index[0].date()),
        "spy_span_end": str(span.index[-1].date()),
        "spy_years": round(n_years, 2),
        "spy_cum_return": spy_cum,
        "spy_cagr": float(spy_cagr),
        "spy_max_dd": spy_mdd,
        "rule_cum_return": rule_cum,
        "rule_max_dd": rule_eq["mdd"],
        "rule_composite": rule_eq["composite"],
    }


# ------- preload OHLC + features for random-MC and expiry scan -------
def preload(tickers):
    loader = DataLoader(cache_dir=str(CONST))
    pipe = FeaturePipeline([TechnicalIndicators(config={
        "rsi_period": 14, "ma_periods": [9, 20, 50], "macd_fast": 12,
        "macd_slow": 26, "macd_signal": 9, "bb_period": 20,
        "volume_ma_period": 20, "consecutive_lookback": 5})])
    data = {}
    for tk in tickers:
        try:
            df = loader.load_ticker(tk, start_date="2014-06-01", end_date=OOS_END)
            if df is None or len(df) < 60:
                continue
            df = pipe.transform(df)
            if "Date" not in df.columns:
                df = df.reset_index()
            df["Date"] = pd.to_datetime(df["Date"])
            df = df.reset_index(drop=True)
            data[tk] = df
        except Exception:
            continue
    return data


def random_entry_mc(trades, data, n_iter=1000, seed=RANDOM_SEED):
    """Same entry dates and per-date counts; random tickers; recompute composite."""
    rng = np.random.default_rng(seed)
    t = trades.sort_values("entry_date")
    counts = t.groupby("entry_date").size()
    # index each ticker by date -> row pos for fast lookup of open & +6 close
    pos = {tk: {d.astype("datetime64[D]"): i for i, d in enumerate(df["Date"].values)}
           for tk, df in data.items()}
    tickers = list(data.keys())

    def trade_ret(tk, entry_date):
        df = data[tk]
        p = pos[tk].get(np.datetime64(pd.Timestamp(entry_date), "D"))
        if p is None or p + HOLD_DAYS >= len(df):
            return None
        o = df["Open"].iloc[p]
        c = df["Close"].iloc[p + HOLD_DAYS]
        if o <= 0:
            return None
        return c / o - 1 - COMMISSION

    obs = metrics(t["net_return"].values)["composite"]
    null, null_ret, null_dd = [], [], []
    for _ in range(n_iter):
        rets = []
        for ed, k in counts.items():
            picks = rng.choice(tickers, size=int(k), replace=False)
            for tk in picks:
                r = trade_ret(tk, ed)
                if r is not None:
                    rets.append(r)
        if len(rets) >= 10:
            m = metrics(rets)
            null.append(m["composite"])
            null_ret.append(float((1 + np.asarray(rets)).prod() - 1))
            null_dd.append(m["mdd"])
    null = np.array(null)
    pct = float((null < obs).mean() * 100)
    p = float((null >= obs).mean())
    np.save(OUT / "random_entry_null.npy", null)
    # save the random-portfolio (cumulative return, max drawdown) cloud for the risk-return scatter
    np.savez(OUT / "random_entry_cloud.npz",
             cum_return=np.array(null_ret), max_dd=np.array(null_dd), composite=null)
    return {"observed": obs, "null_mean": float(null.mean()),
            "null_median": float(np.median(null)), "percentile": pct,
            "p_value": p, "n_iter": int(len(null))}


def expiry_premium(data):
    """Forward 6-day NET return of RSI<22 & >=3 red signals,
    split by signal-day-is-day-before-monthly-expiry vs all other days."""
    exp = pd.to_datetime(ExpiryCalculator.generate_expiry_dates(IS_START, IS_END))
    exp_set = set(np.datetime64(d, "D") for d in exp.values)
    pre, other = [], []
    pre_by_day = {}  # expiry signal_day -> list of returns (for clustering)
    for tk, df in data.items():
        d = df[(df["Date"] >= IS_START) & (df["Date"] <= IS_END)].reset_index(drop=True)
        if "RSI" not in d or "Consecutive_Count" not in d:
            continue
        for i in range(len(d) - HOLD_DAYS - 1):
            if d["RSI"].iloc[i] < RSI_THRESHOLD and d["Consecutive_Count"].iloc[i] >= MIN_CONSECUTIVE:
                o = d["Open"].iloc[i + 1]
                c = d["Close"].iloc[i + 1 + HOLD_DAYS]
                if o <= 0:
                    continue
                ret = c / o - 1 - COMMISSION
                nxt = np.datetime64(pd.Timestamp(d["Date"].iloc[i + 1]), "D")
                if nxt in exp_set:
                    pre.append(ret)
                    pre_by_day.setdefault(nxt, []).append(ret)
                else:
                    other.append(ret)
    pre, other = np.array(pre), np.array(other)
    np.savez(OUT / "expiry_returns.npz", pre=pre, other=other)
    tt = stats.ttest_ind(pre, other, equal_var=False)
    # day-clustered: collapse each expiry day to its mean, test vs non-expiry mean
    day_means = np.array([np.mean(v) for v in pre_by_day.values()])
    tt_clu = stats.ttest_1samp(day_means, other.mean())
    return {"n_pre": int(len(pre)), "n_other": int(len(other)),
            "mean_pre": float(pre.mean()), "mean_other": float(other.mean()),
            "diff": float(pre.mean() - other.mean()),
            "t": float(tt.statistic), "p": float(tt.pvalue),
            "win_pre": float((pre > 0).mean()), "win_other": float((other > 0).mean()),
            "n_expiry_days": int(len(day_means)),
            "day_mean_pre": float(day_means.mean()),
            "cluster_t": float(tt_clu.statistic), "cluster_p": float(tt_clu.pvalue)}


def main():
    tis, toos = get_trades()
    print(f"IS trades={len(tis)}  OOS trades={len(toos)}")
    res = {"is_metrics": metrics(tis["net_return"].values),
           "oos_metrics": metrics(toos["net_return"].values)}
    spy = load_spy()
    print("SPY benchmark ...")
    res["spy"] = spy_benchmark(tis, spy)
    print("preloading OHLC+features ...")
    loader = DataLoader(cache_dir=str(CONST))
    data = preload(loader.get_sp500_tickers())
    print(f"  loaded {len(data)} tickers")
    print("random-entry MC ...")
    res["random_mc"] = random_entry_mc(tis, data)
    print("expiry premium ...")
    res["expiry_premium"] = expiry_premium(data)
    (OUT / "bench_results.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
