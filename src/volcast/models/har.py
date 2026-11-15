"""Heterogeneous AutoRegressive models of realized variance (Corsi, 2009).

    RV[t+1 : t+h] = b0 + b_d * RV_d[t] + b_w * RV_w[t] + b_m * RV_m[t] + e

Fitted per asset by OLS on an expanding window. Variants:
  * levels (the textbook HAR),
  * logs (errors are close to Gaussian, forecasts are always positive),
  * logs + leverage terms (negative past returns raise future variance).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..features import HAR_FEATURES, LEVERAGE_FEATURES


@dataclass
class HAR:
    name: str = "har"
    log: bool = False
    leverage: bool = False
    coefs: dict[str, np.ndarray] = field(default_factory=dict, repr=False)

    def _design(self, df: pd.DataFrame) -> np.ndarray:
        rv = df[HAR_FEATURES].to_numpy(dtype=float)
        cols = [np.ones(len(df)), *(np.log(rv) if self.log else rv).T]
        if self.leverage:
            cols.extend(df[LEVERAGE_FEATURES].to_numpy(dtype=float).T)
        return np.column_stack(cols)

    def fit_predict(self, train: pd.DataFrame, test: pd.DataFrame, horizon: int) -> np.ndarray:
        target = f"target_{horizon}"
        out = np.full(len(test), np.nan)
        test_pos = test.groupby("ticker", sort=False).indices
        for ticker, tr in train.groupby("ticker", sort=False):
            pos = test_pos.get(ticker)
            if pos is None:
                continue
            y = tr[target].to_numpy(dtype=float)
            X = self._design(tr)
            beta, *_ = np.linalg.lstsq(X, np.log(y) if self.log else y, rcond=None)
            pred = self._design(test.iloc[pos]) @ beta
            if self.log:
                # E[exp(x)] = exp(mu + s^2/2) for Gaussian errors: without the
                # correction the forecast targets the median and is biased low.
                resid = np.log(y) - X @ beta
                pred = np.exp(pred + 0.5 * resid.var())
            # "Insanity filter": never forecast outside the range seen in training.
            out[pos] = np.clip(pred, y.min(), y.max())
            self.coefs[ticker] = beta
        return out
