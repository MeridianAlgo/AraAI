#!/usr/bin/env python3
"""
Ara AI Real-Time ML Mode - High-accuracy stock predictions
Primary ML models trained on real market data
"""

import sys
import argparse
import time
from pathlib import Path
import numpy as np
from datetime import timedelta
from functools import lru_cache
try:
    from sklearn.linear_model import (
        Ridge, Lasso, ElasticNet, LinearRegression, HuberRegressor, SGDRegressor, LogisticRegression
    )
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.ensemble import (
        GradientBoostingRegressor, RandomForestRegressor, ExtraTreesRegressor, StackingRegressor
    )
    try:
        from sklearn.ensemble import GradientBoostingClassifier as GBC
    except Exception:
        GBC = None
    try:
        # Some sklearn versions include HistGradientBoostingRegressor
        from sklearn.ensemble import HistGradientBoostingRegressor
    except Exception:
        HistGradientBoostingRegressor = None
    try:
        from sklearn.svm import SVR
    except Exception:
        SVR = None
    SK_AVAILABLE = True
except Exception:
    Ridge = Lasso = ElasticNet = LinearRegression = None
    HuberRegressor = SGDRegressor = LogisticRegression = None
    KNeighborsRegressor = None
    GradientBoostingRegressor = RandomForestRegressor = ExtraTreesRegressor = None
    StackingRegressor = None
    GBC = None
    HistGradientBoostingRegressor = None
    SVR = None
    SK_AVAILABLE = False
try:
    import yfinance as yf
except Exception:
    yf = None

# Optional heavy learners (enabled only if installed and explicitly requested)
try:
    import xgboost as xgb
    from xgboost import XGBRegressor
    XGB_AVAILABLE = True
except Exception:
    XGB_AVAILABLE = False
    XGBRegressor = None
try:
    import lightgbm as lgb
    from lightgbm import LGBMRegressor
    LGB_AVAILABLE = True
except Exception:
    LGB_AVAILABLE = False
    LGBMRegressor = None


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

def main():
    """Real-time ML mode entry point"""
    parser = argparse.ArgumentParser(
        description="Ara AI Real-Time ML - High-accuracy stock predictions",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('symbol', help='Stock symbol (e.g., AAPL, TSLA)')
    parser.add_argument('--days', '-d', type=int, default=5, help='Days to predict (default: 5)')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    parser.add_argument('--retrain', action='store_true', help='Force model retraining')
    parser.add_argument('--period', default='1y', help='Training period (1y, 2y, 5y)')
    parser.add_argument('--ret-cap', type=float, default=0.05, help='Max per-day return cap (fraction, default 0.05)')
    parser.add_argument('--atr-mult', type=float, default=1.5, help='ATR multiple for absolute dollar clamp (default 1.5)')
    parser.add_argument('--no-dollar-clamp', action='store_true', help='Disable absolute $ clamp')
    parser.add_argument('--no-local', action='store_true', help='Disable local ensemble')
    parser.add_argument('--no-news', action='store_true', help='Disable local news sentiment drift')
    parser.add_argument('--no-calibrate', action='store_true', help='Disable volatility calibration and drift adjustment')
    parser.add_argument('--no-rsi', action='store_true', help='Disable RSI gating')
    parser.add_argument('--no-dow', action='store_true', help='Disable Day-of-Week drift')
    parser.add_argument('--no-dom', action='store_true', help='Disable Day-of-Month drift')
    parser.add_argument('--no-dir-gate', action='store_true', help='Disable direction classifier gating')
    parser.add_argument('--no-backtest', action='store_true', help='Skip quick local backtest before prediction')
    parser.add_argument('--backtest-days', type=int, default=30, help='Days to backtest (default: 30)')
    parser.add_argument('--stack', action='store_true', help='Enable stacking meta-learner in local ensemble')
    parser.add_argument('--heavy', action='store_true', help='Enable heavy learners (XGBoost/LightGBM) if installed')
    parser.add_argument('--bands', action='store_true', help='Print uncertainty band width in verbose output')
    parser.add_argument('--framework', choices=['advanced'], default=None, help='Use modular advanced prediction framework')

    args = parser.parse_args()
    ret_cap = float(max(0.01, min(0.2, args.ret_cap)))

    # Optional early path: modular advanced framework
    if getattr(args, 'framework', None) == 'advanced':
        try:
            from meridianalgo.prediction_framework import AdvancedPredictor, FrameworkConfig
            cfg = FrameworkConfig(
                days=int(args.days),
                period=str(args.period),
                use_heavy=bool(args.heavy),
                per_day_cap=float(ret_cap),
                atr_mult=float(args.atr_mult),
                bands=bool(args.bands),
            )
            predictor = AdvancedPredictor(cfg=cfg)
            out = predictor.predict(args.symbol.strip().upper())
            if not out.get('ok'):
                print(f"Advanced framework failed: {out.get('error')}")
                return 1
            cur = float(out.get('current_price') or 0.0)
            prices = list(out.get('predicted_prices') or [])
            day_returns = list(out.get('day_returns') or [])
            cap_used = float(out.get('cap') or ret_cap)
            atr_v = out.get('atr')
            p95 = out.get('p95_abs_move')
            print(f"{args.symbol.strip().upper()} Predictions (Advanced Framework)")
            print(f"ret_cap: ±{cap_used*100:.1f}% | ATR: {0.0 if atr_v is None else atr_v:.2f} | P95$: {0.0 if p95 is None else p95:.2f}")
            print("Day  Price      Change    Return%  Conf%")
            for i, (pr, r) in enumerate(zip(prices, day_returns), start=1):
                try:
                    pr = float(pr); r = float(r)
                except Exception:
                    continue
                change = pr - cur
                conf = max(0.55, 0.8 - 0.05 * (i - 1))
                print(f"{i:>3}  ${pr:>8.2f}  {change:+8.2f}  {r*100:+7.2f}%  {conf*100:6.1f}%")
            if args.bands and isinstance(out.get('bands'), dict):
                bands = out['bands']
                try:
                    low_last = float(bands['low'][-1])
                    high_last = float(bands['high'][-1])
                    last_price = float(prices[-1]) if prices else cur
                    width_pct = (high_last - low_last) / max(1e-9, last_price)
                    print(f"Bands (1σ) final day width: ±{width_pct*50:.2f}%")
                except Exception:
                    pass
            return 0
        except Exception as e:
            print(f"Advanced framework error: {e}")
            return 1

    @lru_cache(maxsize=8)
    def _get_history_cached(symbol: str, period: str):
        """LRU-cached yfinance history fetch (interval=1d)."""
        try:
            if yf is None:
                return None
            return yf.Ticker(symbol).history(period=period or "6mo", interval="1d")
        except Exception:
            return None

    def _local_ts_ensemble(symbol: str, days: int, period: str = "6mo"):
        """Train tiny time-series models on recent data and forecast daily log-returns.
        Returns a list of daily log-returns (len=days) or None on failure.
        """
        if yf is None or not SK_AVAILABLE:
            return None
        try:
            hist = yf.Ticker(symbol).history(period=period or "6mo", interval="1d")
            if hist is None or hist.empty or len(hist) < 60:
                return None
            close = hist['Close'].dropna().values.astype(float)
            # Log returns for stability
            rets = np.diff(np.log(close))
            windows = [10, 20, 40]
            # Ensure enough data for the largest window
            if len(rets) <= max(windows) + 10:
                return None
            # Build broad model set
            base_models = [
                Ridge(alpha=1.0),
                Lasso(alpha=0.0005, max_iter=5000),
                ElasticNet(alpha=0.0005, l1_ratio=0.3, max_iter=5000),
                LinearRegression(),
                HuberRegressor(epsilon=1.5),
                SGDRegressor(max_iter=2000, tol=1e-4, penalty='l2', alpha=1e-4),
                KNeighborsRegressor(n_neighbors=5),
                GradientBoostingRegressor(n_estimators=150, max_depth=3, random_state=42),
                RandomForestRegressor(n_estimators=150, max_depth=6, random_state=42, n_jobs=-1),
                ExtraTreesRegressor(n_estimators=200, max_depth=6, random_state=42, n_jobs=-1)
            ]
            if 'HistGradientBoostingRegressor' in globals() and HistGradientBoostingRegressor is not None:
                base_models.append(HistGradientBoostingRegressor(max_depth=3, max_iter=200, random_state=42))
            if SVR is not None:
                base_models.append(SVR(kernel='rbf', C=1.5, gamma='scale'))
            if getattr(args, 'heavy', False) and 'XGBRegressor' in globals() and XGBRegressor is not None:
                base_models.append(XGBRegressor(n_estimators=200, max_depth=3, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1))
            if getattr(args, 'heavy', False) and 'LGBMRegressor' in globals() and LGBMRegressor is not None:
                base_models.append(LGBMRegressor(n_estimators=300, max_depth=-1, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, random_state=42))

            # Train per-window models and optional meta
            trained_windows = {}
            metas = {}
            for w in windows:
                Xw = []
                yw = []
                for i in range(w, len(rets)):
                    Xw.append(rets[i-w:i])
                    yw.append(rets[i])
                Xw = np.array(Xw)
                yw = np.array(yw)
                if len(yw) < 60:
                    continue
                hold = min(60, max(20, int(0.2 * len(yw))))
                X_train, y_train = Xw[:-hold], yw[:-hold]
                X_hold, y_hold = Xw[-hold:], yw[-hold:]
                fitted = []
                hold_preds = []
                for m in base_models:
                    try:
                        m.fit(X_train, y_train)
                        fitted.append(m)
                        if getattr(args, 'stack', False):
                            try:
                                ph = m.predict(X_hold)
                                hold_preds.append(ph.reshape(-1, 1))
                            except Exception:
                                hold_preds.append(None)
                    except Exception:
                        continue
                if not fitted:
                    continue
                trained_windows[w] = fitted
                if getattr(args, 'stack', False):
                    try:
                        cols = [p for p in hold_preds if p is not None and len(p) == len(y_hold)]
                        if cols and len(cols) >= 3:
                            X_meta = np.hstack(cols)
                            meta = Ridge(alpha=0.5)
                            meta.fit(X_meta, y_hold)
                            metas[w] = meta
                    except Exception:
                        pass

            if not trained_windows:
                return None

            # Multi-step forecasts via iterative updates
            last_wins = {w: rets[-w:].copy() for w in trained_windows.keys()}
            out = []
            for _ in range(days):
                preds_all = []
                for w, models in trained_windows.items():
                    feats = last_wins[w].reshape(1, -1)
                    window_preds = []
                    for m in models:
                        try:
                            p = float(m.predict(feats)[0])
                            p = float(np.clip(p, -0.03, 0.03))
                            window_preds.append(p)
                        except Exception:
                            continue
                    if window_preds:
                        arr = np.array(window_preds, dtype=float)
                        med = float(np.median(arr))
                        arrs = np.sort(arr)
                        k = max(1, int(0.1 * len(arrs))) if len(arrs) > 5 else 0
                        tmean = float(np.mean(arrs[k:-k])) if k > 0 else float(np.mean(arrs))
                        agg = 0.5 * med + 0.5 * tmean
                        if getattr(args, 'stack', False) and w in metas:
                            try:
                                meta_feats = np.array(window_preds, dtype=float).reshape(1, -1)
                                mlen = metas[w].coef_.shape[0] if hasattr(metas[w], 'coef_') else meta_feats.shape[1]
                                if meta_feats.shape[1] == mlen:
                                    agg = float(np.clip(float(metas[w].predict(meta_feats)[0]), -0.03, 0.03))
                            except Exception:
                                pass
                        preds_all.append(agg)
                pred = float(np.median(preds_all)) if preds_all else 0.0
                out.append(pred)
                for w in last_wins:
                    last_wins[w] = np.roll(last_wins[w], -1)
                    last_wins[w][-1] = pred
            return out
        except Exception:
            return None

    def _local_news_sentiment(symbol: str):
        """Read optional local news from data/news/<symbol>.txt and compute a simple sentiment score in [-1,1]."""
        try:
            news_path = Path(__file__).resolve().parents[1] / 'data' / 'news' / f"{symbol.upper()}.txt"
            if not news_path.exists():
                return 0.0
            pos_words = {
                'beat','beats','beating','growth','surge','record','upgrade','upgraded','raise','raised','boost','boosted',
                'strong','outperform','profit','profits','profitable','guidance','exceed','exceeded','win','wins','winning'
            }
            neg_words = {
                'miss','misses','missed','decline','declines','declined','warning','warns','downgrade','downgraded','cut','cuts',
                'probe','investigation','lawsuit','loss','losses','unprofitable','recall','halt','halts','fraud','fine','fined'
            }
            pos = neg = 0
            with open(news_path, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    low = line.strip().lower()
                    if not low:
                        continue
                    for w in pos_words:
                        if w in low:
                            pos += 1
                    for w in neg_words:
                        if w in low:
                            neg += 1
            total = pos + neg
            if total == 0:
                return 0.0
            score = (pos - neg) / total
            # Bound to [-1,1]
            return float(max(-1.0, min(1.0, score)))
        except Exception:
            return 0.0

    def _dow_drift(symbol: str, days: int, period: str = "6mo"):
        """Compute small day-of-week drift adjustments for the next N days (arithmetic returns)."""
        try:
            if yf is None or days <= 0:
                return [0.0] * max(0, days)
            h = _get_history_cached(symbol, period or "6mo")
            if h is None or h.empty or len(h) < 30:
                return [0.0] * days
            idxs = list(h.index)
            closes = h['Close'].to_numpy(dtype=float)
            # Build mean log-return per weekday
            sums = {i: 0.0 for i in range(7)}
            cnts = {i: 0 for i in range(7)}
            for i in range(1, len(closes)):
                prev = float(closes[i-1])
                cur = float(closes[i])
                if prev <= 0:
                    continue
                r = float(np.log(cur / prev))
                dow = int(getattr(idxs[i], 'weekday')())
                sums[dow] += r
                cnts[dow] += 1
            means = {d: (sums[d] / cnts[d]) if cnts[d] > 0 else 0.0 for d in range(7)}
            last_dt = getattr(idxs[-1], 'to_pydatetime')() if hasattr(idxs[-1], 'to_pydatetime') else idxs[-1]
            out = []
            for k in range(1, days + 1):
                dt = last_dt + timedelta(days=k)
                dow = dt.weekday()
                ar = float(np.expm1(means.get(dow, 0.0)))  # convert to arithmetic
                out.append(float(np.clip(ar, -0.002, 0.002)))
            return out
        except Exception:
            return [0.0] * max(0, days)

    def _dom_drift(symbol: str, days: int, period: str = "2y"):
        """Compute small day-of-month drift adjustments for next N days (arithmetic returns)."""
        try:
            if yf is None or days <= 0:
                return [0.0] * max(0, days)
            h = yf.Ticker(symbol).history(period=period or "2y", interval="1d")
            if h is None or h.empty or len(h) < 60:
                return [0.0] * days
            idxs = list(h.index)
            closes = h['Close'].to_numpy(dtype=float)
            # Mean log-return per day of month (1..31)
            sums = {d: 0.0 for d in range(1, 32)}
            cnts = {d: 0 for d in range(1, 32)}
            for i in range(1, len(closes)):
                prev = float(closes[i-1]); cur = float(closes[i])
                if prev <= 0:
                    continue
                r = float(np.log(cur / prev))
                dom = int(getattr(idxs[i], 'day') if hasattr(idxs[i], 'day') else getattr(idxs[i], 'day', 1))
                dom = max(1, min(31, dom))
                sums[dom] += r; cnts[dom] += 1
            means = {d: (sums[d] / cnts[d]) if cnts[d] > 0 else 0.0 for d in range(1, 32)}
            last_dt = getattr(idxs[-1], 'to_pydatetime')() if hasattr(idxs[-1], 'to_pydatetime') else idxs[-1]
            out = []
            for k in range(1, days + 1):
                dt = last_dt + timedelta(days=k)
                dom = max(1, min(31, int(getattr(dt, 'day') if hasattr(dt, 'day') else 1)))
                ar = float(np.expm1(means.get(dom, 0.0)))
                out.append(float(np.clip(ar, -0.002, 0.002)))
            return out
        except Exception:
            return [0.0] * max(0, days)

    def _smooth_seq(vals, w=0.4, cap=0.05):
        """EWMA-like smoothing forward over horizon to reduce unrealistic day-to-day jumps."""
        try:
            if not vals:
                return vals
            out = []
            for i, v in enumerate(vals):
                if i == 0:
                    out.append(float(np.clip(v, -cap, cap)))
                else:
                    sm = (1.0 - w) * out[-1] + w * v
                    out.append(float(np.clip(sm, -cap, cap)))
            return out
        except Exception:
            return vals

    def _dynamic_ret_cap(symbol: str, period: str = "1y", base_cap: float = 0.05):
        """Suggest a dynamic per-day return cap from recent distribution of absolute arithmetic returns."""
        try:
            if yf is None:
                return base_cap
            h = _get_history_cached(symbol, period or "1y")
            if h is None or h.empty or len(h) < 40:
                return base_cap
            close = h['Close'].to_numpy(dtype=float)
            rets_log = np.diff(np.log(close))
            rets_ar = np.expm1(rets_log)
            abs_ar = np.abs(rets_ar)
            p = float(np.percentile(abs_ar, 97.5)) if abs_ar.size > 10 else float(np.max(abs_ar))
            dyn = float(min(0.2, max(0.01, 1.2 * p)))
            return float(min(base_cap, dyn))
        except Exception:
            return base_cap

    def _quick_rsi(symbol: str, period: str = "6mo"):
        """Compute a quick 14-day RSI on close prices. Returns float in [0,100] or None."""
        try:
            if yf is None:
                return None
            h = yf.Ticker(symbol).history(period=period or "6mo", interval="1d")
            if h is None or h.empty or len(h) < 20:
                return None
            closev = h['Close'].to_numpy(dtype=float)
            delta = np.diff(closev)
            if delta.size < 14:
                return None
            gains = np.where(delta > 0, delta, 0.0)
            losses = np.where(delta < 0, -delta, 0.0)
            up = gains[-14:].mean()
            down = losses[-14:].mean()
            rs = up / (down + 1e-9)
            rsi = 100.0 - (100.0 / (1.0 + rs))
            return float(np.clip(rsi, 0.0, 100.0))
        except Exception:
            return None

    def _dir_prob_next(symbol: str, period: str = "6mo", w: int = 20):
        """Estimate next-day probability of up move using a simple classifier on recent returns window."""
        try:
            if yf is None or not SK_AVAILABLE or LogisticRegression is None:
                return None
            h = _get_history_cached(symbol, period or "6mo")
            if h is None or h.empty or len(h) < w + 40:
                return None
            close = h['Close'].to_numpy(dtype=float)
            rets = np.diff(np.log(close))
            if len(rets) <= w + 30:
                return None
            X = []
            y = []
            for i in range(w, len(rets)):
                X.append(rets[i-w:i])
                y.append(1 if rets[i] > 0 else 0)
            X = np.array(X); y = np.array(y)
            # Train/hold split
            hold = min(60, max(20, int(0.2 * len(y))))
            Xtr, ytr = X[:-hold], y[:-hold]
            Xho = X[-hold:]
            # Fit logistic
            clf = LogisticRegression(max_iter=2000, class_weight='balanced')
            clf.fit(Xtr, ytr)
            p_up = float(clf.predict_proba(X[-1].reshape(1, -1))[0,1])
            # Backup: optional GBC
            if (p_up is None or not np.isfinite(p_up)) and GBC is not None:
                try:
                    gbc = GBC(random_state=42)
                    gbc.fit(Xtr, ytr)
                    p_up = float(gbc.predict_proba(X[-1].reshape(1, -1))[0,1])
                except Exception:
                    return None
            if not np.isfinite(p_up):
                return None
            return max(0.0, min(1.0, p_up))
        except Exception:
            return None

    def _atr_caps(symbol: str, period: str = "6mo", n: int = 14):
        """Compute ATR (n-day) and 95th percentile of absolute close-to-close dollar moves."""
        try:
            if yf is None:
                return None, None
            h = yf.Ticker(symbol).history(period=period or "6mo", interval="1d")
            if h is None or h.empty or len(h) < n + 2:
                return None, None
            high = h['High'].to_numpy(dtype=float)
            low = h['Low'].to_numpy(dtype=float)
            close = h['Close'].to_numpy(dtype=float)
            prev_close = np.roll(close, 1)
            tr = np.maximum.reduce([high - low, np.abs(high - prev_close), np.abs(low - prev_close)])
            tr = tr[1:]
            atr_val = float(np.mean(tr[-n:])) if tr.size >= n else None
            abs_moves = np.abs(np.diff(close))
            if abs_moves.size >= 20:
                p95_val = float(np.percentile(abs_moves, 95))
            elif abs_moves.size > 0:
                p95_val = float(np.median(abs_moves))
            else:
                p95_val = None
            if atr_val is not None and not np.isfinite(atr_val):
                atr_val = None
            if p95_val is not None and not np.isfinite(p95_val):
                p95_val = None
            return atr_val, p95_val
        except Exception:
            return None, None

    def _quick_backtest_local(symbol: str, period: str = "1y", days: int = 30):
        """Walk-forward backtest of the local ensemble on last `days` returns. Returns metrics dict."""
        try:
            if yf is None or not SK_AVAILABLE:
                return None
            h = yf.Ticker(symbol).history(period=period or "1y", interval="1d")
            if h is None or h.empty or len(h) < 120:
                return None
            close = h['Close'].to_numpy(dtype=float)
            rets = np.diff(np.log(close))
            T = len(rets)
            windows = [10, 20, 40]
            if T <= max(windows) + days + 10:
                days = max(5, T - (max(windows) + 10))
            if days <= 3:
                return None
            preds = []
            reals = []
            for t in range(T - days, T):
                past = rets[:t]
                if len(past) <= max(windows) + 15:
                    continue
                # Build and fit a small subset of models for speed
                def predict_next_from(past_rets):
                    win_preds = []
                    for w in windows:
                        if len(past_rets) <= w + 20:
                            continue
                        Xw = []
                        yw = []
                        for i in range(w, len(past_rets)):
                            Xw.append(past_rets[i-w:i])
                            yw.append(past_rets[i])
                        Xw = np.array(Xw)
                        yw = np.array(yw)
                        if len(yw) < 50:
                            continue
                        ms = [
                            Ridge(alpha=1.0),
                            GradientBoostingRegressor(n_estimators=120, max_depth=3, random_state=42),
                            RandomForestRegressor(n_estimators=120, max_depth=6, random_state=42, n_jobs=-1)
                        ]
                        preds_w = []
                        for m in ms:
                            try:
                                m.fit(Xw, yw)
                                p = float(m.predict(past_rets[-w:].reshape(1, -1))[0])
                                preds_w.append(float(np.clip(p, -0.03, 0.03)))
                            except Exception:
                                continue
                        if preds_w:
                            med = float(np.median(preds_w))
                            win_preds.append(med)
                    return float(np.median(win_preds)) if win_preds else 0.0
                pred_log = predict_next_from(past)
                preds.append(float(np.expm1(pred_log)))
                reals.append(float(np.expm1(rets[t])))
            if not preds or not reals or len(preds) != len(reals):
                return None
            arr_p = np.array(preds, dtype=float)
            arr_r = np.array(reals, dtype=float)
            mae = float(np.mean(np.abs(arr_p - arr_r)))
            rmse = float(np.sqrt(np.mean((arr_p - arr_r) ** 2)))
            dir_acc = float(np.mean(np.sign(arr_p) == np.sign(arr_r)))
            mape = float(np.mean(np.abs((arr_p - arr_r) / (np.abs(arr_r) + 1e-8))))
            return {"mae": mae, "rmse": rmse, "dir_acc": dir_acc, "mape": mape, "n": len(arr_p)}
        except Exception:
            return None

    try:
        from meridianalgo.ultimate_ml import UltimateStockML
        from meridianalgo.console import ConsoleManager

        target_symbol = args.symbol.strip().upper()

        # Silence heavy rich outputs inside Ultimate modules (keep errors/warnings minimal)
        def _silence_console_manager():
            try:
                heavy = [
                    'print_system_info','print_gpu_info','print_prediction_results','print_ml_predictions',
                    'print_header','print_ultimate_predictions','print_company_analysis','print_ai_company_analysis',
                    'print_accuracy_stats','print_validation_results'
                ]
                for name in heavy:
                    if hasattr(ConsoleManager, name):
                        setattr(ConsoleManager, name, lambda self, *a, **k: None)
                # Keep lightweight text for errors/info
                if hasattr(ConsoleManager, 'print_error'):
                    setattr(ConsoleManager, 'print_error', lambda self, m: print(f"Error: {m}"))
                if hasattr(ConsoleManager, 'print_warning'):
                    setattr(ConsoleManager, 'print_warning', lambda self, m: print(f"Warning: {m}"))
                if hasattr(ConsoleManager, 'print_success'):
                    setattr(ConsoleManager, 'print_success', lambda self, m: print(f"{m}"))
                if hasattr(ConsoleManager, 'print_info'):
                    setattr(ConsoleManager, 'print_info', lambda self, m: print(f"{m}"))
            except Exception:
                pass

        _silence_console_manager()
        console = ConsoleManager(verbose=args.verbose)
        print(f"ARA ULTIMATE ML Analysis - {target_symbol}")

        # Initialize ultimate ML system restricted to the requested symbol
        start_time = time.time()
        ml_system = UltimateStockML()
        ml_system.all_symbols = [target_symbol]
        init_time = time.time() - start_time

        if args.verbose:
            print(f"Ultimate ML system initialized in {init_time:.3f}s")

        # Check training status
        status = ml_system.get_model_status()

        if not status['is_trained'] or args.retrain:
            print(f"Training ULTIMATE ML models on {target_symbol}...")
            training_start = time.time()

            # Train only on the requested symbol
            success = ml_system.train_ultimate_models(
                max_symbols=1,
                period=args.period,
                use_parallel=False
            )
            training_time = time.time() - training_start

            if success:
                print(f"Ultimate models trained in {training_time:.1f}s")

                # Show accuracy scores
                if args.verbose:
                    accuracy_scores = ml_system.accuracy_scores
                    for model, scores in accuracy_scores.items():
                        acc = scores.get('accuracy', 0)
                        print(f"  {model}: {acc:.1f}% accuracy")
            else:
                print("Ultimate training failed")
                return 1

        # Generate ultimate predictions
        print(f"Generating ULTIMATE predictions for {target_symbol}...")

        pred_start = time.time()
        result = ml_system.predict_ultimate(args.symbol, days=args.days)
        pred_time = time.time() - pred_start

        if result:
            # Post-process predictions: remove upward bias and de-compound relative to current price
            current_price = result.get('current_price', 0) or 0.0
            try:
                adjusted = []
                for p in result.get('predictions', []):
                    pr = p.get('predicted_return')
                    price = p.get('predicted_price', current_price)
                    if pr is None and current_price:
                        pr = (price / current_price) - 1.0
                    pr = pr if pr is not None else 0.0
                    pr = max(-ret_cap, min(ret_cap, pr - 0.001))  # clamp and remove slight upward bias
                    new_price = current_price * (1.0 + pr) if current_price else price
                    adjusted.append({
                        'day': p.get('day'),
                        'date': p.get('date'),
                        'predicted_price': float(new_price),
                        'predicted_return': float(pr),
                        'confidence': float(p.get('confidence', 0.7))
                    })
                result['predictions'] = adjusted
            except Exception:
                pass

            # Blend with quick local models (Ridge + GBR etc.) trained on recent data
            local_logrets = None if args.no_local else _local_ts_ensemble(target_symbol, args.days, args.period)
            # Regime metrics for dynamic weighting and calibration
            def _get_regime_metrics(symbol: str, period: str = "6mo"):
                if yf is None:
                    return None
                try:
                    h = yf.Ticker(symbol).history(period=period or "6mo", interval="1d")
                    if h is None or h.empty or len(h) < 40:
                        return None
                    closev = h['Close'].to_numpy(dtype=float)
                    rets = np.diff(np.log(closev))
                    vol = float(np.std(rets))
                    mean = float(np.mean(rets))
                    idx = np.arange(len(closev), dtype=float)
                    c = np.corrcoef(idx, closev)[0,1]
                    trend = float(abs(c)) if np.isfinite(c) else 0.0
                    return {'vol': vol, 'mean': mean, 'trend': trend}
                except Exception:
                    return None

            metrics = _get_regime_metrics(target_symbol, args.period)
            # Dynamic weight: increase with trend, decrease with volatility
            trend = float(metrics.get('trend', 0.0)) if metrics else 0.0
            vol = float(metrics.get('vol', 0.0)) if metrics else 0.0
            # Normalize vol vs a baseline (3% daily log-vol is high); cap influences
            vol_norm = min(1.0, vol / 0.03) if vol > 0 else 0.0
            trend_weight = 0.35 + 0.4 * min(1.0, trend) - 0.2 * vol_norm
            trend_weight = float(max(0.2, min(0.6, trend_weight)))

            # Dynamic return cap from historical distribution
            try:
                dyn_cap = _dynamic_ret_cap(target_symbol, args.period, ret_cap)
                if isinstance(dyn_cap, float) and dyn_cap > 0:
                    ret_cap = float(max(0.01, min(ret_cap, dyn_cap)))
            except Exception:
                pass

            # Local news sentiment adjustment (very small drift)
            news_s = 0.0
            news_drift = 0.0
            if not args.no_news:
                news_s = _local_news_sentiment(target_symbol)
                news_drift = float(np.clip(news_s * 0.0015, -0.003, 0.003))  # <= ±0.3%

            final_preds = []
            final_returns = []
            for i, p in enumerate(result.get('predictions', [])):
                ul_ret = float(p.get('predicted_return', 0.0))
                if local_logrets and i < len(local_logrets):
                    loc_ar = float(np.expm1(local_logrets[i]))  # convert log-return to arithmetic
                    # Penalize disagreement by shrinking magnitude
                    disagree = (np.sign(ul_ret) != np.sign(loc_ar)) and (ul_ret != 0.0 and loc_ar != 0.0)
                    base = (1.0 - trend_weight) * ul_ret + trend_weight * loc_ar
                    if disagree:
                        base *= 0.7
                    ens_ret = base
                else:
                    ens_ret = ul_ret
                # Add tiny news drift
                ens_ret = float(ens_ret + news_drift)
                # Clamp to ±ret_cap
                ens_ret = float(np.clip(ens_ret, -ret_cap, ret_cap))
                final_returns.append(ens_ret)

            # Quick local backtest to calibrate magnitude and confidence
            bt = None if args.no_backtest else _quick_backtest_local(target_symbol, args.period, args.backtest_days)
            # Volatility calibration against realized volatility
            if (not args.no_calibrate) and metrics and final_returns:
                pred_std = float(np.std(final_returns))
                realized_vol = float(metrics.get('vol', 0.0))
                if pred_std > 0 and realized_vol > 0:
                    scale = min(1.0, (realized_vol * 1.25) / pred_std)
                    final_returns = [float(np.clip(r * scale, -ret_cap, ret_cap)) for r in final_returns]
                # Add small drift from realized mean (convert to arithmetic), bounded to ±0.3%
                drift_ar = float(np.expm1(metrics.get('mean', 0.0)))
                drift_ar = float(np.clip(drift_ar, -0.003, 0.003))
                final_returns = [float(np.clip(r + drift_ar, -ret_cap, ret_cap)) for r in final_returns]

            # Use backtest metrics to shrink/expand magnitudes slightly
            if bt and final_returns:
                dir_acc = float(bt.get('dir_acc', 0.5))
                rmse = float(bt.get('rmse', 0.02))
                mag_scale = 0.85 + 0.5 * max(0.0, dir_acc - 0.5)
                mag_scale *= float(max(0.7, min(1.0, ret_cap / max(rmse * 4.0, 1e-6))))
                final_returns = [float(np.clip(r * mag_scale, -ret_cap, ret_cap)) for r in final_returns]

            # Day-of-week drift and RSI gating
            dow_adj = [] if args.no_dow else _dow_drift(target_symbol, args.days, args.period)
            if dow_adj:
                final_returns = [float(np.clip((final_returns[i] if i < len(final_returns) else 0.0) + dow_adj[i], -ret_cap, ret_cap)) for i in range(min(len(dow_adj), len(result.get('predictions', []))))]
            rsi_val = None if args.no_rsi else _quick_rsi(target_symbol, args.period)
            if rsi_val is not None and final_returns:
                new_returns = []
                for r in final_returns:
                    if rsi_val > 70.0 and r > 0:
                        r *= 0.7
                    elif rsi_val < 30.0 and r < 0:
                        r *= 0.7
                    new_returns.append(float(np.clip(r, -ret_cap, ret_cap)))
                final_returns = new_returns

            # Direction gating (shrink when classifier disagrees with sign)
            p_up = None if args.no_dir_gate else _dir_prob_next(target_symbol, args.period)
            if p_up is not None and final_returns:
                gated = []
                for r in final_returns:
                    if r > 0 and p_up < 0.5:
                        r *= 0.7
                    elif r < 0 and p_up > 0.5:
                        r *= 0.7
                    # Confidence scaling of magnitude around certainty
                    scale = 0.9 + 0.2 * abs(p_up - 0.5) / 0.5
                    gated.append(float(np.clip(r * scale, -ret_cap, ret_cap)))
                final_returns = gated

            # Day-of-month drift
            dom_adj = [] if args.no_dom else _dom_drift(target_symbol, args.days, args.period)
            if dom_adj:
                final_returns = [float(np.clip((final_returns[i] if i < len(final_returns) else 0.0) + dom_adj[i], -ret_cap, ret_cap)) for i in range(min(len(dom_adj), len(result.get('predictions', []))))]

            # Smooth day-to-day horizon to reduce oscillations
            final_returns = _smooth_seq(final_returns, w=0.4, cap=ret_cap)

            # Smooth horizon decay (shrink farther days slightly)
            if final_returns:
                decay = np.linspace(1.0, 0.85, len(final_returns))
                final_returns = [float(np.clip(r * decay[i], -ret_cap, ret_cap)) for i, r in enumerate(final_returns)]

            # Compute absolute dollar clamp using ATR and historical absolute moves
            atr_val, p95_move = _atr_caps(target_symbol, args.period)
            abs_cap = None
            if not args.no_dollar_clamp:
                candidates = []
                if atr_val is not None and atr_val > 0:
                    candidates.append(args.atr_mult * atr_val)
                if p95_move is not None and p95_move > 0:
                    candidates.append(1.2 * p95_move)
                if candidates:
                    abs_cap = float(min(candidates))

            # Build final predictions with calibrated returns
            for i, p in enumerate(result.get('predictions', [])):
                ens_ret = final_returns[i] if i < len(final_returns) else float(p.get('predicted_return', 0.0))
                price = current_price * (1.0 + ens_ret) if current_price else p.get('predicted_price', 0.0)
                # Apply absolute dollar clamp around current price
                if abs_cap is not None and current_price:
                    delta = price - current_price
                    if delta > abs_cap:
                        delta = abs_cap
                    elif delta < -abs_cap:
                        delta = -abs_cap
                    price = current_price + delta
                    ens_ret = (price / current_price) - 1.0
                # Confidence adjustment based on ensemble agreement
                base_conf = float(p.get('confidence', 0.7))
                if local_logrets and i < len(local_logrets):
                    loc_ar = float(np.expm1(local_logrets[i]))
                    agree = (np.sign(loc_ar) == np.sign(ens_ret)) or (loc_ar == 0.0 or ens_ret == 0.0)
                    diff = abs(loc_ar - ens_ret)
                    base_conf += 0.05 if agree else -0.10
                    if diff < 0.005:
                        base_conf += 0.05
                # Backtest-informed confidence tweak
                if 'bt' in locals() and bt:
                    base_conf += 0.10 * max(0.0, float(bt.get('dir_acc', 0.5)) - 0.5)
                    # penalize if rmse high relative to cap
                    rmse = float(bt.get('rmse', 0.02))
                    if rmse > 0:
                        base_conf *= float(max(0.8, min(1.0, ret_cap / (rmse * 4.0))))
                conf = float(min(0.95, max(0.4, base_conf)))
                final_preds.append({
                    'day': p.get('day'),
                    'date': p.get('date'),
                    'predicted_price': float(price),
                    'predicted_return': float(ens_ret),
                    'confidence': conf
                })

            # Minimal output (no emojis, simple text)
            symbol = result.get('symbol', target_symbol)
            accuracy = result.get('model_accuracy', 0.0)
            feature_count = result.get('feature_count', 0)
            print(f"{symbol} Predictions (Ultimate) | Acc: {accuracy:.1f}% | Features: {feature_count}")
            if args.verbose:
                rsi_txt = f", RSI: {rsi_val:.1f}" if rsi_val is not None else ""
                dow_mean = float(np.mean(dow_adj)) if dow_adj else 0.0
                extra = ""
                try:
                    if 'p_up' in locals() and p_up is not None:
                        extra = f" | p_up={p_up:.3f}"
                except Exception:
                    pass
                print(f"Blend weight (local): {trend_weight:.2f} | News drift: {news_drift:+.4f} | DOW mean: {dow_mean:+.4f}{rsi_txt} | ret_cap: ±{ret_cap*100:.1f}% | stack: {getattr(args,'stack',False)} | heavy: {getattr(args,'heavy',False)} | horizon_decay_min=0.85{extra}")
                if 'bt' in locals() and bt:
                    print(f"Backtest [{int(bt.get('n',0))}d]: dir_acc={bt.get('dir_acc',0.0)*100:.1f}% | MAE={bt.get('mae',0.0):.4f} | RMSE={bt.get('rmse',0.0):.4f} | MAPE={bt.get('mape',0.0):.2f}")
                if args.bands:
                    band_rmse = float(bt.get('rmse', 0.0)) if 'bt' in locals() and bt else 0.0
                    band_vol = float(np.expm1(metrics.get('vol', 0.0))) if metrics else 0.0
                    band_w = float(min(ret_cap, max(0.005, 1.5 * max(band_rmse, band_vol))))
                    print(f"Uncertainty band (ret): ±{band_w*100:.2f}%")
                if 'abs_cap' in locals() and abs_cap is not None:
                    print(f"Abs dollar clamp: ±${abs_cap:.2f} (atr_mult={args.atr_mult})")
                else:
                    print("Abs dollar clamp: disabled")
            print("Day  Date        Price     Change    Return%  Conf%")
            to_print = final_preds if final_preds else result.get('predictions', [])
            for p in to_print:
                day = p.get('day', 0)
                date = (p.get('date') or '')[:10]
                price = p.get('predicted_price', 0.0)
                change = price - (current_price or price)
                ret_pct = p.get('predicted_return', 0.0) * 100.0
                conf_pct = p.get('confidence', 0.0) * 100.0
                print(f"{day:>3}  {date:<10}  ${price:>8.2f}  {change:+8.2f}  {ret_pct:+7.2f}%  {conf_pct:6.1f}%")

            total_time = time.time() - start_time
            print(f"Ultimate analysis completed in {total_time:.2f}s (prediction: {pred_time:.3f}s)")

            # Show comprehensive performance
            accuracy = result.get('model_accuracy', 0)
            feature_count = result.get('feature_count', 0)
            print(f"Model Accuracy: {accuracy:.1f}% | Features: {feature_count} | Models: 8")

            # Show market status
            market_status = result.get('market_status', {})
            if market_status.get('is_open'):
                print("Market is currently OPEN")
            else:
                print("Market is currently CLOSED")

            # Show sector info
            sector_info = result.get('sector_info', {})
            if sector_info.get('sector') != 'Unknown':
                print(f"Sector: {sector_info['sector']} | Industry: {sector_info['industry']}")

        else:
            print(f"Ultimate prediction failed for {args.symbol}")
            return 1

        return 0

    except KeyboardInterrupt:
        print("\nCancelled by user")
        return 1
    except Exception as e:
        print(f"Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())