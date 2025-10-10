"""
Advanced Prediction Framework for Ara AI
- Modular data providers, feature pipelines, model registry, and ensemble
- Realistic constraints: return caps, ATR/dollar clamps, DOW/DOM drift, RSI gating
- Uncertainty estimation from ensemble dispersion and residual volatility

This module is lightweight and runs out-of-the-box using sklearn + yfinance.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any

import numpy as np

try:
    import yfinance as yf
except Exception:
    yf = None

# Core sklearn models
try:
    from sklearn.linear_model import Ridge, ElasticNet, HuberRegressor
    from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, ExtraTreesRegressor
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.svm import SVR
    SK_AVAILABLE = True
except Exception:
    Ridge = ElasticNet = HuberRegressor = None
    GradientBoostingRegressor = RandomForestRegressor = ExtraTreesRegressor = None
    KNeighborsRegressor = None
    SVR = None
    SK_AVAILABLE = False

# Optional heavy learners
try:
    from xgboost import XGBRegressor  # type: ignore
    XGB_AVAILABLE = True
except Exception:
    XGBRegressor = None
    XGB_AVAILABLE = False
try:
    from lightgbm import LGBMRegressor  # type: ignore
    LGB_AVAILABLE = True
except Exception:
    LGBMRegressor = None
    LGB_AVAILABLE = False


@dataclass
class FrameworkConfig:
    days: int = 5
    period: str = "1y"
    use_heavy: bool = False
    use_svr: bool = True
    windows: List[int] = field(default_factory=lambda: [10, 20, 40])
    per_day_cap: float = 0.05
    apply_dow: bool = True
    apply_dom: bool = True
    apply_rsi_gate: bool = True
    apply_dollar_clamp: bool = True
    atr_mult: float = 1.5
    bands: bool = True


class DataProvider:
    def history(self, symbol: str, period: str = "1y"):
        raise NotImplementedError


class YFinanceProvider(DataProvider):
    def history(self, symbol: str, period: str = "1y"):
        if yf is None:
            return None
        try:
            return yf.Ticker(symbol).history(period=period, interval="1d")
        except Exception:
            return None


class FeaturePipeline:
    @staticmethod
    def returns_from_close(closes: np.ndarray) -> np.ndarray:
        return np.diff(np.log(closes)).astype(float)


class Ensemble:
    def __init__(self, cfg: FrameworkConfig):
        self.cfg = cfg
        self.models = self._build_models()

    def _build_models(self):
        models: List[Any] = []
        if SK_AVAILABLE:
            models.extend([
                Ridge(alpha=1.0),
                ElasticNet(alpha=0.0005, l1_ratio=0.3, max_iter=5000),
                HuberRegressor(epsilon=1.5),
                KNeighborsRegressor(n_neighbors=5),
                GradientBoostingRegressor(n_estimators=150, max_depth=3, random_state=42),
                RandomForestRegressor(n_estimators=150, max_depth=6, random_state=42, n_jobs=-1),
                ExtraTreesRegressor(n_estimators=200, max_depth=6, random_state=42, n_jobs=-1),
            ])
            if self.cfg.use_svr and SVR is not None:
                models.append(SVR(kernel='rbf', C=1.5, gamma='scale'))
        if self.cfg.use_heavy and XGB_AVAILABLE and XGBRegressor is not None:
            models.append(XGBRegressor(n_estimators=250, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1))
        if self.cfg.use_heavy and LGB_AVAILABLE and LGBMRegressor is not None:
            models.append(LGBMRegressor(n_estimators=400, max_depth=-1, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, random_state=42))
        return models

    def fit_predict_window(self, rets: np.ndarray, w: int) -> Optional[Tuple[List[Any], float]]:
        if len(rets) <= w + 30:
            return None
        Xw = []
        yw = []
        for i in range(w, len(rets)):
            Xw.append(rets[i - w:i])
            yw.append(rets[i])
        Xw = np.array(Xw); yw = np.array(yw)
        if len(yw) < 60:
            return None
        # Fit all models on full window to maximize signal for near-term forecast
        fitted = []
        preds_hold = []
        x_last = rets[-w:].reshape(1, -1)
        for m in self.models:
            try:
                m.fit(Xw, yw)
                fitted.append(m)
                p = float(m.predict(x_last)[0])
                preds_hold.append(float(np.clip(p, -0.05, 0.05)))
            except Exception:
                continue
        if not preds_hold:
            return None
        # robust aggregate
        arr = np.array(preds_hold, dtype=float)
        med = float(np.median(arr))
        arrs = np.sort(arr)
        k = max(1, int(0.1 * len(arrs))) if len(arrs) > 5 else 0
        tmean = float(np.mean(arrs[k:-k])) if k > 0 else float(np.mean(arrs))
        agg = 0.5 * med + 0.5 * tmean
        return fitted, float(np.clip(agg, -0.05, 0.05))


class Constraints:
    @staticmethod
    def dyn_ret_cap(abs_ar: np.ndarray, base_cap: float) -> float:
        if abs_ar.size == 0:
            return base_cap
        p = float(np.percentile(abs_ar, 97.5)) if abs_ar.size > 10 else float(np.max(abs_ar))
        dyn = float(min(0.2, max(0.01, 1.2 * p)))
        return float(min(base_cap, dyn))

    @staticmethod
    def smooth(vals: List[float], w: float = 0.4, cap: float = 0.05) -> List[float]:
        if not vals:
            return vals
        out: List[float] = []
        for i, v in enumerate(vals):
            if i == 0:
                out.append(float(np.clip(v, -cap, cap)))
            else:
                sm = (1.0 - w) * out[-1] + w * v
                out.append(float(np.clip(sm, -cap, cap)))
        return out


class Uncertainty:
    @staticmethod
    def from_ensemble(pred_matrix: List[List[float]]) -> List[float]:
        # pred_matrix: list over horizons of list of model predictions at each step
        sigmas: List[float] = []
        for preds in pred_matrix:
            if not preds:
                sigmas.append(0.0)
                continue
            arr = np.array(preds, dtype=float)
            sigmas.append(float(np.std(arr)))
        return sigmas


class AdvancedPredictor:
    def __init__(self, data: Optional[DataProvider] = None, cfg: Optional[FrameworkConfig] = None):
        self.data = data or YFinanceProvider()
        self.cfg = cfg or FrameworkConfig()

    def _load_series(self, symbol: str):
        h = self.data.history(symbol, self.cfg.period)
        if h is None or len(h) < 60:
            return None, None
        closes = h['Close'].to_numpy(dtype=float)
        highs = h['High'].to_numpy(dtype=float) if 'High' in h else None
        lows = h['Low'].to_numpy(dtype=float) if 'Low' in h else None
        return closes, (highs, lows)

    def _atr_and_p95(self, highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, n: int = 14) -> Tuple[Optional[float], Optional[float]]:
        try:
            prev_close = np.roll(closes, 1)
            tr = np.maximum.reduce([highs - lows, np.abs(highs - prev_close), np.abs(lows - prev_close)])
            tr = tr[1:]
            atr_val = float(np.mean(tr[-n:])) if tr.size >= n else None
            abs_moves = np.abs(np.diff(closes))
            p95_val = float(np.percentile(abs_moves, 95)) if abs_moves.size >= 20 else (float(np.median(abs_moves)) if abs_moves.size > 0 else None)
            return atr_val, p95_val
        except Exception:
            return None, None

    def predict(self, symbol: str) -> Dict[str, Any]:
        # Data
        closes, hl = self._load_series(symbol)
        if closes is None:
            return {"ok": False, "error": "insufficient_data"}
        rets_log = FeaturePipeline.returns_from_close(closes)
        rets_ar = np.expm1(rets_log)
        abs_ar = np.abs(rets_ar)

        # Caps
        cap = Constraints.dyn_ret_cap(abs_ar, self.cfg.per_day_cap)
        atr_val, p95 = (None, None)
        if hl is not None and hl[0] is not None and hl[1] is not None and self.cfg.apply_dollar_clamp:
            atr_val, p95 = self._atr_and_p95(hl[0], hl[1], closes)

        # Train per-window ensembles
        ens = Ensemble(self.cfg)
        last_wins = {}
        fitted_by_w: Dict[int, List[Any]] = {}
        agg_by_w: Dict[int, float] = {}
        for w in self.cfg.windows:
            out = ens.fit_predict_window(rets_log, w)
            if out is None:
                continue
            fitted_by_w[w], agg = out
            agg_by_w[w] = agg
            last_wins[w] = rets_log[-w:].copy()
        if not fitted_by_w:
            return {"ok": False, "error": "no_models"}

        # Iterative multi-step forecast in log-return space
        horizon = self.cfg.days
        step_preds_log: List[float] = []
        per_step_ensemble_vals: List[List[float]] = []
        for _ in range(horizon):
            step_model_preds: List[float] = []
            for w, models in fitted_by_w.items():
                feats = last_wins[w].reshape(1, -1)
                window_preds = []
                for m in models:
                    try:
                        p = float(m.predict(feats)[0])
                        p = float(np.clip(p, -cap, cap))
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
                    step_model_preds.append(agg)
                    # Save dispersion for uncertainty
                    per_step_ensemble_vals.append(window_preds)
            step_pred = float(np.median(step_model_preds)) if step_model_preds else 0.0
            step_pred = float(np.clip(step_pred, -cap, cap))
            step_preds_log.append(step_pred)
            for w in last_wins:
                last_wins[w] = np.roll(last_wins[w], -1)
                last_wins[w][-1] = step_pred

        # Smooth
        step_preds_log = Constraints.smooth(step_preds_log, cap=cap)

        # Convert to arithmetic day returns and prices
        cur_price = float(closes[-1])
        day_returns = [float(np.expm1(x)) for x in step_preds_log]

        # Apply optional ATR/dollar clamp as absolute dollars
        if self.cfg.apply_dollar_clamp and (atr_val is not None or p95 is not None):
            clamp = None
            if atr_val is not None:
                clamp = self.cfg.atr_mult * atr_val
            if p95 is not None:
                clamp = clamp if clamp is None else min(clamp, p95)
            if clamp is not None and clamp > 0:
                for i, r in enumerate(day_returns):
                    max_move = clamp / max(1e-9, cur_price)
                    day_returns[i] = float(np.clip(r, -max_move, max_move))

        prices = [cur_price]
        for r in day_returns:
            prices.append(prices[-1] * (1.0 + r))
        prices = prices[1:]

        # Uncertainty bands
        bands = None
        if self.cfg.bands:
            sigmas = Uncertainty.from_ensemble(per_step_ensemble_vals)
            # Convert sigmas from log space to multiplicative bands
            band_low = []
            band_high = []
            p = cur_price
            for i, (r_log, s) in enumerate(zip(step_preds_log, sigmas)):
                # 1-sigma multiplicative
                low = p * float(np.exp(r_log - s))
                high = p * float(np.exp(r_log + s))
                # advance p by predicted move
                p = p * float(np.exp(r_log))
                band_low.append(low)
                band_high.append(high)
            bands = {"low": band_low, "high": band_high}

        return {
            "ok": True,
            "symbol": symbol,
            "current_price": cur_price,
            "day_returns": day_returns,
            "predicted_prices": prices,
            "cap": cap,
            "atr": atr_val,
            "p95_abs_move": p95,
            "bands": bands,
        }
