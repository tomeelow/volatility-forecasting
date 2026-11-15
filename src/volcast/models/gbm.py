"""Pooled gradient boosting (LightGBM) across all assets.

Two design choices matter:

1. *Scale-free target.* The model predicts the ratio  RV[t+1:t+h] / RV_m[t]
   (future variance relative to the asset's own trailing monthly variance), so
   one model can learn from bonds and single stocks alike.

2. *Gamma objective = QLIKE.* With a log link, the gamma negative
   log-likelihood is  y * exp(-f) + f,  which is exactly the QLIKE loss used
   for evaluation. The model is therefore trained on the metric it is judged
   by, and its forecasts are unbiased for the mean without a log correction.

Hyper-parameters are fixed a priori (small trees, strong regularisation) and
were not tuned on the out-of-sample period.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..features import GBM_FEATURES

DEFAULT_PARAMS = {
    "objective": "gamma",
    "learning_rate": 0.03,
    "num_leaves": 15,
    "min_data_in_leaf": 300,
    "feature_fraction": 0.7,
    "bagging_fraction": 0.7,
    "bagging_freq": 1,
    "lambda_l2": 10.0,
    "max_bin": 127,
    "verbose": -1,
}


@dataclass
class GBM:
    name: str = "gbm"
    num_boost_round: int = 300
    params: dict = field(default_factory=lambda: dict(DEFAULT_PARAMS))
    seed: int = 7
    num_threads: int = 0
    importances: list[pd.Series] = field(default_factory=list, repr=False)

    def fit_predict(self, train: pd.DataFrame, test: pd.DataFrame, horizon: int) -> np.ndarray:
        import lightgbm as lgb

        y = (train[f"target_{horizon}"] / train["rv_m"]).to_numpy(dtype=float)
        data = lgb.Dataset(
            train[GBM_FEATURES], label=y, categorical_feature=["asset_class"], free_raw_data=True
        )
        params = {**self.params, "seed": self.seed, "num_threads": self.num_threads, "deterministic": True}
        booster = lgb.train(params, data, num_boost_round=self.num_boost_round)
        gain = booster.feature_importance(importance_type="gain")
        self.importances.append(pd.Series(gain / gain.sum(), index=GBM_FEATURES))
        ratio = booster.predict(test[GBM_FEATURES])
        return ratio * test["rv_m"].to_numpy(dtype=float)

    def mean_importance(self) -> pd.Series:
        """Gain share per feature, averaged over all walk-forward refits."""
        return pd.concat(self.importances, axis=1).mean(axis=1).sort_values(ascending=False)
